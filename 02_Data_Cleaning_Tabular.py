from __future__ import division
import pandas as pd
import numpy as np
import geopandas as gpd

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from utility import printHeader
from utility import asPercentage

# Our plots often refer to yellow/green, I have chosen hex-codes which match the official colours of NYC taxis.
colorscheme = ["#8db600", "#fce300"]


# This generator automatically gives IDs to axes, useful for the large number of subplots in Figures 3 and 4.
def generateAxes(rows, cols):
    id = 0
    while id < rows * cols:
        yield_x = id // cols
        yield_y = id % cols
        yield (yield_x, yield_y)
        id += 1


################################################
# Loading the dataset and creating new features
################################################

printHeader("Loading the dataset and creating new features")

# Load the data from the pickle file we created
combined = pd.read_pickle("data/combined_pre_cleaning.pickle")
print ("Loaded")

# Calculate a new feature - the tip percentage
combined["tip_percentage"] = 100 * combined["tip_amount"] / (combined["total_amount"])

# Create a duration feature, first converting the datetimes to pandas datetime format
combined["dropoff_datetime"] = pd.to_datetime(combined["dropoff_datetime"])
combined["pickup_datetime"] = pd.to_datetime(combined["pickup_datetime"])
combined["duration"] = (combined["dropoff_datetime"] - combined["pickup_datetime"]) \
    .apply(lambda td: td.total_seconds())


# Create a month feature, showing which month the data came from (using the filename)
def getMonthFromFilename(filename):
    return filename.split("/")[1].split("_")[-1].split("-")[1].split(".")[0]


combined["month"] = combined["filename"].apply(getMonthFromFilename)

# Total Number of Journeys in the dataset
totalJourneys = len(combined)
print "There are %d trips in the dataset" % totalJourneys

# We sample the full dataset for the plots to speed up processing
combinedSample = combined.sample(n=10000)

################################################
# Producing Figures 3
################################################

printHeader("Figure 3")

axIDs = generateAxes(3, 3)

# We create a new figure with 9 subplots
fig, ax = plt.subplots(3, 3, figsize=(3 * 5, 4 * 5))

### Month Counts ###
axID = axIDs.next()
combined["month"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Month")
ax[axID].set_xlabel("Month")
ax[axID].set_ylabel("Number of Trips")

### Ratecode ID ###
axID = axIDs.next()
combinedSample["RatecodeID"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Ratecode")
ax[axID].set_xlabel("Ratecode ID")
ax[axID].set_ylabel("Number of Trips")

### Vendor ID ###
axID = axIDs.next()
combinedSample["VendorID"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Vendor ID")
ax[axID].set_xlabel("Vendor ID")
ax[axID].set_ylabel("Number of Trips")

### Passenger Count ###
axID = axIDs.next()
combinedSample["passenger_count"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Passenger Count")
ax[axID].set_xlabel("Number of Passengers")
ax[axID].set_ylabel("Number of Trips")

### Payment Type ###
axID = axIDs.next()
combinedSample["payment_type"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Payment Type")
ax[axID].set_xlabel("Payment Type")
ax[axID].set_ylabel("Number of Trips")

### Store and Forward Flag ###
axID = axIDs.next()
combinedSample["store_and_fwd_flag"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Store and Forward Flag")
ax[axID].set_xlabel("Store and Forward Flag")
ax[axID].set_ylabel("Number of Trips")

### Cab Colour ###
axID = axIDs.next()
combinedSample["colour"].value_counts().sort_index().plot(ax=ax[axID], kind="bar", color="black")
ax[axID].set_title("Yellow. vs Green Cabs")
ax[axID].set_xlabel("Colour")
ax[axID].set_ylabel("Number of Trips")

### Tip Amount ###
axID = axIDs.next()
combinedSample["tip_amount"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Tip Amount")
ax[axID].set_xlabel("USD")

### Tip Percentage ###
axID = axIDs.next()
combinedSample["tip_percentage"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Tip Percentage")
ax[axID].set_xlabel("%")

# Set the tight layout flag and save the plot
plt.tight_layout()
plt.savefig("figures/fig3.png")
print "Saved fig3.png.\n"
plt.clf()

################################################
# Producing Figure 4
################################################

printHeader("Figure 4")

# Reuse our axes ID generator
axIDs = generateAxes(3, 3)

# Create a new 3x3 figure of subplots
fig, ax = plt.subplots(3, 3, figsize=(3 * 5, 4 * 5))

### Tolls Amount ###
axID = axIDs.next()
combinedSample["tolls_amount"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Tolls Amount")
ax[axID].set_xlabel("USD")

### Fare Amount ###
axID = axIDs.next()
combinedSample["fare_amount"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Fare Amount")
ax[axID].set_xlabel("USD")

### Improvement Surcharge ###
axID = axIDs.next()
combinedSample["improvement_surcharge"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Improvement Surcharge")
ax[axID].set_xlabel("USD")

### MTA Tax ###
axID = axIDs.next()
combinedSample["mta_tax"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("MTA Tax")
ax[axID].set_xlabel("USD")

### Extra ###
axID = axIDs.next()
combinedSample["extra"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Extra")
ax[axID].set_xlabel("USD")

### Total Amount ###
axID = axIDs.next()
combinedSample["total_amount"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Total Amount")
ax[axID].set_xlabel("USD")

### Duration ###
axID = axIDs.next()
combinedSample["duration"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Duration")
ax[axID].set_xlabel("Minutes")

### Distance ###
axID = axIDs.next()
combinedSample["trip_distance"].plot(ax=ax[axID], kind="kde", color="black")
ax[axID].set_title("Distance")
ax[axID].set_xlabel("Miles")

# Set the tight layout flag and show the plot
plt.tight_layout()
plt.savefig("figures/fig4.png")
print "Saved fig 4.png.\n"
plt.clf()

################################################
# Inspecting the payment mechanism feature
################################################
printHeader("Payment Mechanism")

combined.groupby("payment_type").mean()["tip_percentage"].plot(kind="bar", color="black")
plt.title("Average Tip Percentage by Payment Type")
plt.xlabel("Payment Type")
plt.ylabel("Mean Tip Percentage")
plt.clf()
# Note this figure isn't saved as it isn't referenced in the report, but it's how we know that the non-credit card
# trips don't have tip data

################################################
# Figure 6 - Distribution of Number of Passengers
################################################
printHeader("Figure 6 - Distribution of Number of Passengers")

# Plot the overall distribution of passengers
groupedBySourceAndPCount = combined.groupby(["passenger_count"])["tip_percentage"]
groupedBySourceAndPCount.count().plot(kind="bar", color="black", figsize=(10, 10))
plt.xlabel("Number of Passengers")
plt.ylabel("Count of Trips")

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig6.png")
print "Saved fig6.png.\n"
plt.clf()

# Calculate the percentage that are within the expected range
percentageAccepted = np.sum(groupedBySourceAndPCount.count()[1:7]) / np.sum(groupedBySourceAndPCount.count())
print "Percentage of journeys within bounds: %s" % asPercentage(percentageAccepted)
print "Percentage rejected: %s" % asPercentage(1 - percentageAccepted)

################################################
# Pickup and Dropoff Timestamps Check
################################################
printHeader("Pickup and Dropoff Timestamps check")
# To identify datetimes that do not fit in to the 2018, we have to be careful because journeys can begin in one year
# and end in another. To account for this I add a one day tolerance.

print "There are %s trips taking place prior to 2018" % sum(
    (combined["pickup_datetime"] < "2017-12-31") | (combined["dropoff_datetime"] < "2018-01-01"))
print "There are %s trips taking place after 2018" % sum(
    (combined["pickup_datetime"] >= "2019-01-01") | (combined["dropoff_datetime"] >= "2019-01-02"))

################################################
# Figure 7 - Distribution of Fares
################################################
printHeader("Figure 7 - Distribution of Fares")

# Create a KDE plot of the fare amount feature
base = combinedSample["fare_amount"].plot(kind="kde", color="black", figsize=(10, 5))

# plt.xlim(-100,200) # Switching this will zoom in on the main section but you won't see the full tails
# Add axes labels and title
plt.xlabel("Fare")
plt.title("Distribution of fares")

# Plot the total amount on the same plot
combinedSample["total_amount"].plot(ax=base, kind="kde", color="red", linestyle="--")

# Add a legend
plt.legend()

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig7.png")
print "Saved fig7.png.\n"
plt.clf()

# We calculate and print some summary statistics for the feature
countNegativeFares = np.sum(combined["total_amount"] < 0)
countZeroFares = np.sum(combined["total_amount"] == 0)
countLowFares = np.sum(combined["total_amount"] <= 1)
countHighFares = np.sum(combined["total_amount"] > 60)

print "Count of negative fares: %d. As a percentage of total journeys: %s" % (
    countNegativeFares, asPercentage(countNegativeFares / len(combined)))
print "Count of zero fares: %d. As a percentage of total journeys: %s" % (
    countZeroFares, asPercentage(countZeroFares / len(combined)))
print "Count of low fares: %d. As a percentage of total journeys: %s" % (
    countLowFares, asPercentage(countLowFares / len(combined)))
print "Count of high fares: %d. As a percentage of total journeys: %s" % (
    countHighFares, asPercentage(countHighFares / len(combined)))
print "The lowest fare is: $%f" % combined["total_amount"].min()
print "The highest fare is: $%f" % combined["total_amount"].max()

################################################
# Negative Fares
################################################

# # We load the entire June yellow dataset, rather than the mixed sample we've used so far and split it in to
# positive and negative fares # This is because we want to match negative fares against their theoretical positive
# twin.
combined_june = pd.read_csv("data/yellow_tripdata_2018-06.csv")
combined_june["dropoff_datetime"] = pd.to_datetime(combined_june["tpep_dropoff_datetime"])
combined_june["pickup_datetime"] = pd.to_datetime(combined_june["tpep_pickup_datetime"])

negativeFares = combined_june[combined_june["fare_amount"] < 0]
positiveFares = combined_june[combined_june["fare_amount"] >= 0]


# This function will take a row and search for trips that have the same value for 5 key features. It will return the
# number of such matching trips.
def calculateMatchingRecords(row):
    DOLocationID = row["DOLocationID"]
    PULocationID = row["PULocationID"]
    dropoff_datetime = row["dropoff_datetime"]
    pickup_datetime = row["pickup_datetime"]
    fare_amount = row["fare_amount"]

    # We query the positiveFares dataframe for records with the same pickup and dropoff locations and datetimes as
    # the row passed, but with the negative of the fare. We then return the number of matching records from the
    # positiveFares dataframe.
    numberOfMatches = sum(
        (positiveFares["DOLocationID"] == DOLocationID) & (positiveFares["PULocationID"] == PULocationID) & (
                positiveFares["dropoff_datetime"] == dropoff_datetime) & (
                positiveFares["pickup_datetime"] == pickup_datetime) & (
                positiveFares["fare_amount"] == -fare_amount))
    return numberOfMatches


# We run this matching function on a sample of the negative fares. We only use a sample because it's slow to run but
# this sample can be increased to increase robustness.
negativeFares_sample = negativeFares.sample(n=100)
negativeFares_sample["matchesInPositive"] = negativeFares_sample.apply(calculateMatchingRecords, axis=1)

# Having calculated how many matches in the positive dataset we have, we see how many values we have for the number
# of matches
print negativeFares_sample["matchesInPositive"].value_counts()

################################################
# Negative Fares
################################################
printHeader("Figure 8 - Negative Fares - Payment Types")

# Create a figure to plot on
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot a bar showing negative fares with matches vs. those without
negativeFares_sample["matchesInPositive"].value_counts().plot(ax=ax[0], color="black", kind="bar")
ax[0].set_xticklabels(["Match", "No Match"])

# Plot the payment type distribution for all trips
combined_june.groupby("payment_type").count()["fare_amount"].plot(ax=ax[1], color="black", kind="bar")
ax[1].set_xlabel("Payment Type")

# Plot a bar showing payment type distribution for negative fares
negativeFares.groupby("payment_type").count()["fare_amount"].plot(ax=ax[2], color="black", kind="bar")
ax[2].set_xlabel("Payment Type")

plt.tight_layout()
plt.savefig("figures/fig8.png")
print "Saved fig8.png.\n"
plt.clf()

################################################
# Very High Fares
################################################
printHeader("Very High Fares")
# We look up the trip with the highest fare. We actually have to reset the index as because we combined the
# dataframes, the index isn't really unique.
print "Highest Fare in the dataset:"
print combined.reset_index().loc[combined.reset_index()["total_amount"].idxmax()]

################################################
# Figure 9 - Distribution of duration
################################################
printHeader("Figure 9 - Distribution of duration")

# We calculate some summary statistics for the duration feature and print them out
countNegativeDurations = np.sum(combined["duration"] < 0)
countZeroDurations = np.sum(combined["duration"] == 0)
countShortDurations = np.sum(combined["duration"] <= 60)
countLongDurations = np.sum(combined["duration"] > 60 * 60)

print "Count of negative durations: %d" % countNegativeDurations
print "Count of zero durations: %d. As a percentage of total journeys: %s" % (
countZeroDurations, asPercentage(countZeroDurations / totalJourneys))
print "Count of short durations: %d. As a percentage of total journeys: %s" % (
countShortDurations, asPercentage(countShortDurations / totalJourneys))
print "Count of long durations: %d. As a percentage of total journeys: %s" % (
countLongDurations, asPercentage(countLongDurations / totalJourneys))
print "The longest journey is: %d hours" % int(combined["duration"].max() / 60 / 60)

# We plot a KDE of the sample duration feature
combinedSample = combined.sample(n=100000)
base = combinedSample["duration"].plot(kind="kde", color="black", figsize=(10, 5))

# Add axes, title and legend
plt.xlabel("Duration")
plt.title("Distribution of Trip Durations")
plt.legend()

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig9.png")
print "Saved fig9.png.\n"
plt.clf()

# Print a list of dates associated with negative durations
# This will show they occur on the day the clocks change.
printHeader("We review the dates associated with journeys of negative duration to check if they correspond with "
            "certain events")

# We look at the journeys with negative duration and review the distribution of of dates associated with these journeys
print "Date and count of journeys with negative duration"
print combined[combined["duration"] < 0]["dropoff_datetime"].dt.date.value_counts()

################################################
# Figure 10 - Trip Distance Feature
################################################
printHeader("Figure 10 - Trip Distance Feature")

# We calculate some summary statistics fro the distance feature and print them out
countNegativeDistances = np.sum(combined["trip_distance"] < 0)
countZeroDistances = np.sum(combined["trip_distance"] == 0)
countLowDistances = np.sum(combined["trip_distance"] <= 0.1)
countHighDistances = np.sum(combined["trip_distance"] > 50)

print "Count of negative distances: %d. As a percentage of total journeys: %s" % (
countNegativeDistances, asPercentage(countNegativeDistances / totalJourneys))
print "Count of zero distances: %d. As a percentage of total journeys: %s" % (
countZeroDistances, asPercentage(countZeroDistances / totalJourneys))
print "Count of short distances: %d. As a percentage of total journeys: %s" % (
countLowDistances, asPercentage(countLowDistances / totalJourneys))
print "Count of long distances: %d. As a percentage of total journeys: %s" % (
countHighDistances, asPercentage(countHighDistances / totalJourneys))
print "The longest trip is: %f miles" % combined["trip_distance"].max()

# Plot the distance feature
combinedSample["trip_distance"].plot(kind="kde", color="black", figsize=(15, 5))

# Tidy up axes
plt.xlabel("Distance")
plt.xlim(0)

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig10.png")
print "Saved fig10.png.\n"
plt.clf()

################################################
# Figure 11 - Trips > 50 miles
################################################
printHeader("Figure 11 - Trips greater than 50 miles that start or end outside of NYC")

# Load the spatial data, calculate centroids for each zone and drop all other data from the geodataframe
gdf = gpd.read_file("spatialref/taxi_zones.shp").set_index("OBJECTID")
gdf["centroids"] = gdf["geometry"].apply(lambda geometry: geometry.centroid)
gdf["x"] = gdf["centroids"].apply(lambda centroid: centroid.coords[0][0])
gdf["y"] = gdf["centroids"].apply(lambda centroid: centroid.coords[0][1])
spatial_ref_data = gdf[["x", "y"]]

# Join the centroid coordinates we calculated to the tabular data
combined = combined.join(spatial_ref_data.rename(columns={"x": "dropoff_x", "y": "dropoff_y"}), on="DOLocationID")
combined = combined.join(spatial_ref_data.rename(columns={"x": "pickup_x", "y": "pickup_y"}), on="PULocationID")

# We define a long trip as any trip over 50 miles
longTrips = combined[combined["trip_distance"] > 50]

# Set up a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot for the main populations
# n.b we know if a journey started/ended outside of NYC because the pickup_x or dropoff_x values are NA.
combined[["pickup_x", "dropoff_x"]].isna().any(axis=1).value_counts().plot(ax=ax[0], kind="bar", color="black")

# Tidy up axes. We use a custom formatter to make the y axis more readable.
ax[0].set_title("All trips")
ax[0].set_xticklabels(["False", "True"], rotation=0)
ax[0].set_xlabel("Starts or ends outside of NYC")
ax[0].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Plot the same features for long trips only
# n.b we know if a journey started/ended outside of NYC because the pickup_x or dropoff_x values are NA.
longTrips[["pickup_x", "dropoff_x"]].isna().any(axis=1).value_counts().plot(ax=ax[1], kind="bar", color="black")
ax[1].set_title("Trips greater than 50 miles")
ax[1].set_xticklabels(["False", "True"], rotation=0)
ax[1].set_xlabel("Starts or ends outside of NYC")

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig11.png")
print "Saved fig11.png.\n"
plt.clf()

################################################
# Figure 12 - Rate code distribution for trips greater than 50 miles vs. all trips
################################################
printHeader("Figure 12 - Rate code distribution for trips greater than 50 miles vs. all trips")

# Create a figure to plot on
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot the ratecode distribution for all trips, set title and axis label
combined["RatecodeID"].value_counts().sort_index().plot(ax=ax[0], kind="bar", color="black")
ax[0].set_title("All trips")
ax[0].set_xlabel("Rate Code")

# Plot the ratecode distribution for long trips only, set title and axis label
longTrips["RatecodeID"].value_counts().sort_index().plot(ax=ax[1], kind="bar", color="black")
ax[1].set_title("Trips greater than 50 miles")
ax[1].set_xlabel("Rate Code")

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig12.png")
print "Saved fig12.png.\n"
plt.clf()

################################################
# Figure 13 - Pair plots of distance, duration and total fare
################################################
printHeader("Figure 13 - Pair plots of distance, duration and total fare")

# We create a new figure with three subplots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Plot duration vs. total amount
ax[0].scatter(combinedSample["duration"], combinedSample["total_amount"], color="black", alpha=0.1)

# Tidy the axes and set a title
ax[0].set_xlabel("Trip Duration (seconds)")
ax[0].set_ylabel("Trip Cost USD")
ax[0].set_title("Cost vs. Duration")

# Plot duration vs. trip distance
ax[1].scatter(combinedSample["duration"], combinedSample["trip_distance"], color="black", alpha=0.1)

# Tidy the axes and set a title
ax[1].set_xlabel("Trip Duration (seconds)")
ax[1].set_ylabel("Trip Distance (miles)")
ax[1].set_title("Distance vs. Duration")

# Plot distance vs. total amount
ax[2].scatter(combinedSample["trip_distance"], combinedSample["total_amount"], color="black", alpha=0.1)

# Tidy the axes and set a title
ax[2].set_xlabel("Trip Distance (miles)")
ax[2].set_ylabel("Trip Cost USD")
ax[2].set_title("Cost vs. Distance")

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig13.png")
print "Saved fig13.png.\n"
plt.clf()

################################################
# Figure 14 - Clustering
################################################
printHeader("Figure 14 - Clustering")

# We run a k-mean clustering model with k = 2
km = KMeans(n_clusters=2).fit(combinedSample[["duration", "total_amount"]])

# Create a figure on which to plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Get the nearest cluster for each example in the sample
combinedSample["cluster"] = km.predict(combinedSample[["duration", "total_amount"]])

# Plot the data again showing which data we keep
ax[0].scatter(combinedSample["duration"], combinedSample["total_amount"],
              c=combinedSample["cluster"].apply(lambda cluster: "green" if cluster == 0 else "red"), alpha=0.1)

# Plot the kmeans centroids
for cc in km.cluster_centers_:
    ax[0].scatter(cc[0], cc[1], c="black")

# Add title and axes labels
ax[0].set_title("Data with k-means model (k=2) applied.")
ax[0].set_xlabel("Duration")
ax[0].set_ylabel("USD")

# We create a custom legend and add it to the graph
legend_elements = [
    Line2D([0], [0], marker='o', color='g', label="Data to be retained", markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='r', label="Data to be excluded", markerfacecolor='red', markersize=10)]
ax[0].legend(handles=legend_elements)

# We drop the data that's in the second cluster
combinedSample = combinedSample[combinedSample["cluster"] == 0]

# We plot the data again, without the dropped data
ax[1].scatter(combinedSample["duration"], combinedSample["total_amount"], color="black", alpha=0.1)

# Add title and axes labels
ax[1].set_title("Data after cleaning.")
ax[1].set_xlabel("Duration")
ax[1].set_ylabel("USD")

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig14.png")
print "Saved fig14.png.\n"
plt.clf()

# We calculate how much data we have retained through this process
print "Data retained: %.2f%%" % (100 * (1 - combinedSample["cluster"]).sum() / len(combinedSample))

################################################
# Figure 15 - Linear Regression
################################################
printHeader("Figure 15 - Linear Regression")
# We run a linear regression to identify a line of best fit
LR = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
LR.fit(combinedSample[["trip_distance"]], combinedSample["total_amount"])

# Create a figure on which to plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# We plot the original fare against distance plot
ax[0].scatter(combinedSample["trip_distance"], combinedSample["total_amount"], color="black", alpha=0.1)

# We plot the line of best fit, along with parallel lines of best fit at + and - $50
X = np.arange(80)
Y = LR.predict(X.reshape(-1, 1))
ax[0].plot(X, Y)
ax[0].plot(X, Y + 75, c="red")
ax[0].plot(X, Y - 75, c="red")

# Add title and axes labels
ax[0].set_title("Total Cost of Journey vs. Distance with Linear Regression")
ax[0].set_xlabel("Distance")
ax[0].set_ylabel("USD")

# We store the length of the dataframe before cleaning so we can estimate how much data is dropped
sizeBeforeCleaning = len(combinedSample)

# Filter out trips beyond the tolerance
tolerance = 75 # Tolerance was discovered through visual analytics
combinedSample = combinedSample[ \
    (combinedSample["total_amount"] < LR.predict(combinedSample[["trip_distance"]]) + 75) & \
    (combinedSample["total_amount"] > LR.predict(combinedSample[["trip_distance"]]) - 75)]

# We plot the new sample without the data which was dropped
ax[1].scatter(combinedSample["trip_distance"], combinedSample["total_amount"], color="black", alpha=0.1)

# We plot the line of best fit, along with parallel lines of best fit at + and - $75
X = np.arange(80)
Y = LR.predict(X.reshape(-1, 1))
ax[1].plot(X, Y)
ax[1].plot(X, Y + 75, c="red")
ax[1].plot(X, Y - 75, c="red")

# Add title and axes labels
ax[1].set_title("Data after cleaning.")
ax[1].set_xlabel("Distance")
ax[1].set_ylabel("USD")

# Set the tight_layout flag and save the figure.
plt.tight_layout()
plt.savefig("figures/fig15.png")
print "Saved.\n"
plt.clf()

# We calculate how much data we have retained through this process
print "Data retained: %.2f%%" % (100 * len(combinedSample) / sizeBeforeCleaning)

###########################
# Unreferenced Location IDs
###########################
printHeader("Unreferenced Location IDs")

# Create a set of the unique locations appearing in the pickup and dropoff features
uniqueLocations = set(combined["PULocationID"].unique()) or set(combined["DOLocationID"].unique())

# Create a list of the locations IDs that aren't in the spatial reference data and print them out
unreferencedLocations = [location for location in uniqueLocations if location not in gdf.index]
print "The following locations appear in the tabular data but not in the spatial reference data:"
for location in unreferencedLocations: print "* %s" % location

###########################
# Data Cleaning Summary
###########################
printHeader("Data Cleaning Summary")

# We apply all the filtering operations to clean the data.
# We store the size of the dataframe in a list after each major operation so we can track the impact of each.
cleaning_record = []

def updateCleaningRecord(action):
    cleaning_record.append([action, len(combined)])
    print "%s. %s trips." % (action, len(combined))

updateCleaningRecord("Prior to cleaning")

# Remove non-credit card journeys
combined = combined[combined["payment_type"] == 1]
updateCleaningRecord("Removed non-credit card trips")

# Remove trips with passenger count of 0 or greater than 6
combined = combined[(combined["passenger_count"] >= 1) & (combined["passenger_count"] <= 6)]
combined = combined[combined["duration"] >= 0]
updateCleaningRecord("Removed spurious passenger count trips")

# Remove trips with negative duration
combined = combined[combined["duration"] > 0]
updateCleaningRecord("Removed trips with negative duration")

# Remove trips that have spurious timestamps
combined = combined[(combined["pickup_datetime"] >= "2017-12-31") & (combined["dropoff_datetime"] >= "2018-01-01")]
combined = combined[(combined["pickup_datetime"] < "2019-01-01") & (combined["dropoff_datetime"] < "2019-01-02")]
updateCleaningRecord("Removed trips with datetimes outside of range")

# Remove trips with locations outside of the known zones Location IDs 264 and 265 don't appear in the spatial
# reference data. There is no mention of them in the data documentation.
combined = combined[
    (~combined["PULocationID"].isin(unreferencedLocations)) & (~combined["DOLocationID"].isin(unreferencedLocations))]
updateCleaningRecord("Removed trips to undefined zone locations")

# Remove negative fare or zero fare
combined = combined[combined["total_amount"] > 0]
updateCleaningRecord("Removed trips with a negative or zero fare")

# Remove trips with a fare beyond the tolerance predicted given the distance
combined = combined[ \
    (combined["total_amount"] < LR.predict(combined[["trip_distance"]]) + tolerance) & \
    (combined["total_amount"] > LR.predict(combined[["trip_distance"]]) - tolerance)]
updateCleaningRecord("Removed trips with a fare beyond the tolerance predicted given the distance")

# Remove trips with high duration but low fare using our clustering approach
combined["cluster"] = km.predict(combined[["duration", "total_amount"]])
combined = combined[combined["cluster"] == 0]
updateCleaningRecord("Removed trips with high duration but low fare using clustering approach")

# At each stage we recorded the total number of rows in the dataframe.
# We want to show the relative change after each stage so we subtract the consecutive amounts to calculate the delta.
cleaning_record_delta = [[cleaning_record[i][0], cleaning_record[i - 1][1] - cleaning_record[i][1]] for i in
                         range(1, len(cleaning_record))]

# We reverse the order of the list so the plot appears in the correct order
cleaning_record_delta.reverse()

# We plot the deltas as a bar chart
summaryFigure = pd.DataFrame(cleaning_record_delta).set_index(0).plot(kind="barh", figsize=(12, 5), legend=False,
                                                                      color="black")

# We remove the y axis label
plt.ylabel("")

# We set the tight layout flag and save the figure
plt.tight_layout()
plt.savefig("figures/fig16.png")
print "Saved fig16.png.\n"
plt.clf()

###########################
# Save cleaned data
###########################
printHeader("Save cleaned data")
combined.to_pickle("data/combined_post_cleaning_trips.pickle")
print "Saved to data/combined_post_cleaning_trips.pickle"
