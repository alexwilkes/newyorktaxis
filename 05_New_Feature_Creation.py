from __future__ import division
import pandas as pd
from utility import printHeader

###########################
# Loading cleaned data
###########################
printHeader("Loading cleaned data")

# We load the three pickles we created during cleaning
combined = pd.read_pickle("data/combined_post_cleaning_trips.pickle")
weather = pd.read_pickle("data/weather_2018.pickle").sort_index()
spatial_ref_data = pd.read_pickle("data/spatial.pickle")
print "Loaded."


###########################
# Creating new features
###########################
printHeader("Creating new features")
# This function will classify examples as either '0' which corresponds to a tip percentage greater than 0.1% or 1
# otherwise
def classifyTipPercentage(tip_percentage):
    if tip_percentage < 0.1 and tip_percentage >= 0:
        return 1
    elif tip_percentage >= 0.1:
        return 0
    else:
        return 2

# We apply our classifyTipPercentage feature to generate a new feature
combined["zero_tip"] = combined["tip_percentage"].apply(classifyTipPercentage)

# This function will convert the colour of the cab in to a numeric representation
def colourAsFactor(source):
    if source == "yellow":
        return 0
    elif source == "green":
        return 1
    else:
        return 2

# We convert the colour in to the numeric representation
combined["colour"] = combined["colour"].apply(colourAsFactor)

# We convert the ratecode and colour features to one hot encoded features
combined = pd.concat([
    combined,
    pd.get_dummies(combined['RatecodeID'], prefix='Ratecode'),
    pd.get_dummies(combined['colour'], prefix='colour')],
    axis=1)

###########################
# Temporal Features
###########################

# We add two new sets of features, a month and a day of week - both are created as dummy features
combined = pd.concat([
    combined,
    pd.get_dummies(combined["dropoff_datetime"].dt.dayofweek, prefix="temporal_dayofweek"),
    pd.get_dummies(combined["dropoff_datetime"].dt.month, prefix="temporal_month")],
    axis=1)

# This function will take a time in pandas datetime format and return the number of minutes elapsed since midnight
def convertToMinutes(time):
    return time.hour * 60 + time.minute

# We create a new feature with the above function
combined["temporal_timeInMinutes"] = combined["dropoff_datetime"].dt.time.apply(convertToMinutes)

###########################
# Joining Weather Data
###########################

# We will join the weather using the pickup_datetime of the journey as the key on which to join.
# We have to convert this field to a pandas datetime and also sort it for the join to succeed.
combined["pickup_datetime"] = pd.to_datetime(combined["pickup_datetime"])
combined.sort_values(inplace=True, by="pickup_datetime")

# We create a new dataframe by joining the two
combined = pd.merge_asof(combined, weather, left_on="pickup_datetime", right_index=True)

###########################
# Dropoff/pickup borough
###########################
# We lookup the borough of the pickup and the dropoff and add these as new features

# Add the pickup borough
combined = pd.merge(
    combined,
    spatial_ref_data[["borough"]],
    how="left",
    left_on="PULocationID",
    right_index=True
)

# Rename the new feature we created so it's clear it's the pickup
combined.rename(columns={"borough": "pickup_borough"}, inplace=True)

# Add the dropoff borough
combined = pd.merge(
    combined,
    spatial_ref_data[["borough"]],
    how="left",
    left_on="DOLocationID",
    right_index=True
)

# Rename the new feature we created so it's clear it's the dropoff
combined.rename(columns={"borough": "dropoff_borough"}, inplace=True)

# We encode the boroughs we just joined in to one hot encoded variables
combined = pd.concat([
    combined,
    pd.get_dummies(combined["pickup_borough"], prefix="pickup_borough"),
    pd.get_dummies(combined["dropoff_borough"], prefix="dropoff_borough")],
    axis=1)

###########################
# Saving data with additional features
###########################
printHeader("Saving data with additional features")
combined.to_pickle("data/combined_post_new_features.pickle")
print "Saved to data/combined_post_new_features.pickle"