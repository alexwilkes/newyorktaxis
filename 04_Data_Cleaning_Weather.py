from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

from utility import printHeader

###########################
# Loading the data
###########################
printHeader("Loading the data")
# We load the csv in to a dataframe. We supply the date parser to pandas, as otherwise pandas assumes American
# date formats.
weather = pd.read_csv("data/Weather_extract_2018.csv", sep=",", index_col=0, parse_dates=True,
                      date_parser=lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M'))

# We replace 'Trace of precipitation' with a very low value as discussed in the report, just so we can plot it.
precipitation_clean = weather["RRR"].dropna().replace("Trace of precipitation", 0.00001).astype(float)

print "Weather Data Loaded."

###########################
# Plot the two weather features
###########################
printHeader("Plotting the two weather features")

# Create subplots
fig, ax = plt.subplots(2, 1, figsize=(12, 12))

# Plot the temperature feature on the first subplot
weather["T"].plot(ax=ax[0], kind="line", color="black")
ax[0].set_title("Temperature")
ax[0].set_ylabel("c")

# We have to manually apply the month labels to the x-axis
ax[0].set_xticks(["2018-%s-01" % month for month in range(1, 13)])
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax[0].set_xticklabels(["01 %s" % month for month in months])
ax[0].set_xlabel("")

# Plot the precipitation feature on the second subplot.
precipitation_clean.plot(ax=ax[1], kind="line", color="black")
ax[1].set_title("Precipitation")
ax[1].set_ylabel("mm")

# Again we have to apply the month labels to the y-axis
ax[1].set_xticks(["2018-%s-01" % month for month in range(1, 13)])
ax[1].set_xticklabels(["01 %s" % month for month in months])
ax[1].set_xlabel("")

# Set the tight layout flag and save the plot
plt.tight_layout()
plt.savefig("figures/fig17.png")
print "Fig 17 Saved.\n"
plt.clf()

###########################
# Checking for missing data
###########################
printHeader("Checking for missing data")
print "Missing Data in temperature feature:"
print weather[weather["T"].isna()]

print "Replacing missing value with known temperature"
weather.at["2018-04-18 23:00:00"] = 9

print "\nMissing Data in precipitation feature:"
print weather[(weather["RRR"].isna()) | (weather["RRR"] == "Trace of precipitation")]["RRR"].head(10)

print "Replacing missing value and 'trace of precipitation' with relevant values"
# We update the precipitation feature, converting NaNs to 0s, replacing 'Trace of precipitation' with a small
# numerical value and casting the feature from a string to a float.
weather["RRR"] = weather["RRR"] \
    .fillna(0) \
    .replace("Trace of precipitation", 0.00001) \
    .astype(float)

###########################
# Saving weather data
###########################
printHeader("Saving weather data")
weather[["T", "RRR"]].to_pickle("data/weather_2018.pickle")
print "Saved weather data to data/weather_2018.pickle"