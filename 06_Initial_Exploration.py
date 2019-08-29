from __future__ import division
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.mixture import GaussianMixture
from utility import printHeader

import math
import scipy.stats as st

import matplotlib.pyplot as plt

# Our plots often refer to yellow/green, I have chosen hex-codes which match the official colours of NYC taxis.
colorscheme = ["#fce300", "#8db600"]

###########################
# Loading cleaned data with new features
###########################
printHeader("Loading cleaned data with new features")

# Load all of the data and combine it in to a single data frame
combined = pd.read_pickle("data/combined_post_new_features.pickle")
print "Tabular Data Loaded."

# Read the shape files in to a geopandas dataframe
spatial = gpd.read_file("spatialref/taxi_zones.shp").set_index("LocationID")
print "Spatial Data Loaded"

###########################
# Figure 18 - Mixture Model with Distribution of Tip Percentage
###########################
printHeader("Figure 18 - Mixture Model with Distribution of Tip Percentage")

# We take a sample of the dataset for the mixture model to speed up processing
sample = combined["tip_percentage"].sample(100000)

# We cut off the extremes of the distribution as we're trying to find the main peaks in the middle
sample = sample[sample > -1]
sample = sample[sample < 30]

# Create the GMM model - we have to reshape the data to avoid an SKLearn warning about 1 dimensional feature vectors
gmm = GaussianMixture(n_components=6) \
    .fit(np.array(sample).reshape(-1, 1))

# Plot the underlying distribution using a kernel density estimate
base = sample.plot(kind="kde", color="black")

# Iterate through the means in the fitted GMM, print each one and plot it as a vertical dashed lines
print "GMM component means:"
for mean in sorted(gmm.means_):
    base.axvline(x=mean, c="red", linestyle="--")
    print "* $%.2f" % float(mean)

# Tidy up the axes, set the tight layout flag and save the plot
plt.xlim([-5, 30])
plt.xlabel("Tipping Percentage")
plt.tight_layout()
plt.savefig("figures/fig18.png")
print "Saved fig18.png.\n"
plt.clf()

###########################
# Figure 19 - Tip Percentage vs. Cab Colour
###########################
printHeader("Figure 19  - Tip Percentage vs. Cab Colour")

# Create a figure to plot on
fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# Plot the mean tip % grouped by passenger count
combined.groupby(["colour"])["tip_percentage"].mean().plot(ax=ax[0], kind="bar", color=colorscheme, figsize=(18, 5))
ax[0].set_xlabel("Cab Colour")
ax[0].set_ylabel("Average Tip Percentage")

# Plot the mean tip % grouped by passenger count
combined[combined["colour"] == 0]["tip_percentage"].plot(ax=ax[1], kind="hist", orientation="horizontal", bins=100,
                                                         density=True, color=colorscheme[0])

# Tidy up axes
ax[1].set_ylim([-5, 40])
ax[1].set_xlim([0, 0.7])
ax[1].set_xlabel("Average Tip Percentage")

# Plot the mean tip % grouped by passenger count
combined[combined["colour"] == 1]["tip_percentage"].plot(ax=ax[2], kind="hist", orientation="horizontal", bins=100,
                                                         density=True, color=colorscheme[1])

# Tidy up axes
ax[2].set_ylim([-5, 40])
ax[2].set_xlim([0, 0.7])
ax[2].set_xlabel("Average Tip Percentage")

# Set the tight layout flag and save the figure
plt.tight_layout()
plt.savefig("figures/fig19.png")
print "Saved fig19.png.\n"
plt.clf()

###########################
# Figure 20 - Tip Percentage vs. Passenger Count
###########################
printHeader("Figure 20 - Tip Percentage vs. Passenger Count")

# Plot the data by the passenger count feature
groupedBySourceAndPCount = combined.groupby(["passenger_count"])["tip_percentage"]

# Plot the mean tip % grouped by passenger count
groupedBySourceAndPCount.mean().plot(kind="bar", color="black", figsize=(10, 10))

# Tidy up the axes, set the tight layout flag and save the plot
plt.xlabel("Passenger Count")
plt.ylabel("Mean Tip Percentage")
plt.tight_layout()
plt.savefig("figures/fig20.png")
print "Saved fig20.png.\n"
plt.clf()

###########################
# Figure 21 - Temporal Features
###########################
printHeader("Figure 21 - Temporal Features")

# Create figure to plot on
fig, ax = plt.subplots(2, 2, figsize=(16, 16))

# First two plots - number of trips by day of the Week

# In order to ensure the plot is sorted by days of the week as we expect, we have to explicitly set the sorting order
#  and assign this order to the summary table. We create a dictionary which maps days of week to an order
days_of_week = dict(zip(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], range(7)))

# We count the number of trips for each day of the week and store the result
summary_count = combined.groupby(by=[combined.pickup_datetime.dt.weekday_name]).count()
# We use the dictionary to store the number associated with the day of the week against the counts
summary_count["day_of_week_id"] = summary_count.index.map(lambda i: days_of_week[i])

# We plot the results, sorted by the number assigned from the dictionary
summary_count.sort_values(by="day_of_week_id")["tip_percentage"] \
    .plot(ax=ax[0, 0], kind="bar", color="black")

# Label the plot
ax[0, 0].set_xlabel("Day of Week")
ax[0, 0].set_ylabel("Number of Journeys")

# Number of trips by hour
summary_count = combined.groupby(by=[combined.pickup_datetime.dt.hour]).count()
summary_count["tip_percentage"].plot(ax=ax[0, 1], kind="bar", color="black")

# Tidy up the axes
ax[0, 1].set_xlabel("Hour")
ax[0, 1].set_ylabel("Number of Journeys")

# Second two plots - Mean Tip Percentage

# As before, but with means instead of counts, we create a summary stat for each day of the week and add the ordering
summary_mean = combined.groupby(by=[combined.pickup_datetime.dt.weekday_name]).mean()
summary_mean["day_of_week_id"] = summary_mean.index.map(lambda i: days_of_week[i])

# We plot the summary means, sorted by the ordering we created
summary_mean.sort_values(by="day_of_week_id")["tip_percentage"] \
    .plot(ax=ax[1, 0], kind="bar", color="black")

# Label the axes
ax[1, 0].set_xlabel("Day of Week")
ax[1, 0].set_ylabel("Mean Tip Percentage")

# Finally we do the same thing but for mean tip percentage, grouped by hour of the day
summary_mean = combined.groupby(by=[combined.pickup_datetime.dt.hour]).mean()
summary_mean["tip_percentage"].plot(ax=ax[1, 1], kind="bar", color="black")

# Label the axes
ax[1, 1].set_xlabel("Hour")
ax[1, 1].set_ylabel("Average Tip Percentage")

# Toggle on tight layout and save the plots
plt.tight_layout()
plt.savefig("figures/fig21.png")
print "Saved fig21.png.\n"
plt.clf()

###########################
# Figure 22 - Spatial Features
###########################
printHeader("Figure 22 - Spatial Features")

# Create a series indexed by zone with the mean tip % for pickups in each zone
tipPercentageAveragesByZone_pickup = combined.groupby("PULocationID").mean()["tip_percentage"]
tipPercentageAveragesByZone_pickup.name = "tip_percentage_pickup"  # Rename the series to tip_percentage_pickup

# Create a series indexed by zone with the mean tip % for dropoffs in each zone
tipPercentageAveragesByZone_dropoff = combined.groupby("DOLocationID").mean()["tip_percentage"]
tipPercentageAveragesByZone_dropoff.name = "tip_percentage_dropoff"  # Rename the series to tip_percentage_pickup

# Create a series indexed by zone with the count of pickups in each zone
countJourneysByZone_pickup = combined.groupby("PULocationID").count()["tip_percentage"]
countJourneysByZone_pickup.name = "count_pickup"  # Rename the series to count_pickup

# Create a series indexed by zone with the count of dropoffs in each zone
countJourneysByZone_dropoff = combined.groupby("DOLocationID").count()["tip_percentage"]
countJourneysByZone_dropoff.name = "count_dropoff"  # Rename the series to count_dropoff

# We apply a number of join transformations to add the features of interest to each location in the geopandas dataframe
spatial = spatial \
    .join(tipPercentageAveragesByZone_pickup) \
    .join(countJourneysByZone_pickup) \
    .join(tipPercentageAveragesByZone_dropoff) \
    .join(countJourneysByZone_dropoff)

# We create a separate geodataframe that contains the zones we don't want to plot with colour. We don't want to plot
# them with colour because either they don't appear in the tabular data, or the numberof journeys is too few. Instead
# we plot them as grey.
excludeFromMainPlot = (spatial["count_pickup"] < 20) | (spatial["count_dropoff"] < 20) | \
                      (spatial[['tip_percentage_dropoff', 'tip_percentage_pickup']].isna().any(axis=1))

# We create separate dataframes for the zones we plot with colour and those we plot with grey
spatialExcluded = spatial[excludeFromMainPlot]
spatial = spatial[excludeFromMainPlot == False]

# Plot four maps. Two for pickups, two for drop-offs. Two showing the number of journeys in each zone,
# the other showing average tip.
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# Plot a map of geometries in the two dataframes for number of pickups
spatial.plot(ax=axes[0, 0], column="count_pickup", legend=True)
spatialExcluded.plot(ax=axes[0, 0], color="#DCDCDC")

# Tidy the axes on the plot
axes[0, 0].set_title("Map showing count of trips by pickup zone")
axes[0, 0].set_xticks([])  # Get rid of the axis ticks
axes[0, 0].set_yticks([])

# Plot a map of geometries in the two dataframes for number of dropoffs
spatial.plot(ax=axes[0, 1], column="count_dropoff", legend=True)
spatialExcluded.plot(ax=axes[0, 1], color="#DCDCDC")

# Tidy the axes on the plot
axes[0, 1].set_title("Map showing count of trips by dropoff zone")
axes[0, 1].set_xticks([])  # Get rid of the axis ticks
axes[0, 1].set_yticks([])

# Plot a map of geometries in the two dataframes for mean tip percentage for pickups
spatial.plot(ax=axes[1, 0], column="tip_percentage_pickup", legend=True)
spatialExcluded.plot(ax=axes[1, 0], color="#DCDCDC")

# Tidy the axes on the plot
axes[1, 0].set_title("Map showing average tip percentage by pickup zone")
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])

# Plot a map of geometries in the two dataframes for mean tip percentage for dropoffs
spatial.plot(ax=axes[1, 1], column="tip_percentage_dropoff", legend=True)
spatialExcluded.plot(ax=axes[1, 1], color="#DCDCDC")

# Tidy the axes on the plot
axes[1, 1].set_title("Map showing average tip percentage by dropoff zone")
axes[1, 1].set_xticks([])  # Get rid of the axis ticks
axes[1, 1].set_yticks([])

# Set tight layout flag. Save the figure
plt.tight_layout()
plt.savefig("figures/fig22.png")
print "Saved fig22.png.\n"
plt.clf()

###########################
# Figure 23 - Weather
###########################
printHeader("Figure 23 - Weather")

# We define bins we want to cut the data in to, for precipitation and temperature
p_bins = [-10, 0, 10, 20, 30, 40, 50]
t_bins = [-20, -10, 0, 10, 20, 30, 40, ]

# We identify the bin each trip falls in to, for temperature and for precipitation, and record this as a new column
combined["precipitation_bin"] = pd.cut(combined["RRR"], p_bins).apply(lambda bin: "%s - %s" % (bin.left, bin.right))
combined["temperature_bin"] = pd.cut(combined["T"], t_bins).apply(lambda bin: "%s - %s" % (bin.left, bin.right))

# Create a plot to plot on
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the Tip Percentage By Temperature Bin
combined.groupby("temperature_bin").mean()["tip_percentage"] \
    .plot(ax=ax[0], kind="bar", color="black")

# Label the first plot
ax[0].set_xlabel("Temperature Bin")
ax[0].set_ylabel("Mean Tip Percentage")

# Plot the Tip Percentage By Precipitation Bin
combined.groupby("precipitation_bin").mean()["tip_percentage"] \
    .plot(ax=ax[1], kind="bar", color="black")

# Label the first plot
ax[1].set_xlabel("Precipitation Bin")
ax[1].set_ylabel("Mean Tip Percentage")

# Toggle on tight layout and save the plots
plt.tight_layout()
plt.savefig("figures/fig23.png")
print "Saved fig23.png.\n"
plt.clf()

# Because it's not obvious from the visuals, we do a hypothesis test to see if the tipping percentage is different
# across the weather features

# This function is defined for undertaking difference of two means hypothesis tests.
# It takes the sample means, the size of the two samples, the standard deviations and the desired level.
# It will print the key information as the hypothesis is undertaken and return the final decision.
def hypothesisTestTwoProportions_twoSided_t(sampleMeans, sampleN, sampleStdDeviations, alpha):
    # The standard error is calculated
    # SE = sqrt(s1^2/n1 + s2^2/n2)
    SE = math.sqrt((sampleStdDeviations[1] ** 2) / sampleN[1] + (sampleStdDeviations[0] ** 2) / sampleN[0])
    print "Standard error: %s" % SE

    # The z-statistic is calculated by taking the difference of the two sample proportions.
    # It's then divided by the standard error
    t = (sampleMeans[1] - sampleMeans[0]) / SE
    print "t-statistic: %s" % t

    # We calculate the z-value at which we would reject. It's a two sided test so we divide in two.
    df = sampleN.sum() - 1
    print "DF: %s" % df
    threshold = st.t.ppf(1 - alpha / 2, df)
    print "Threshold for rejection: %s" % threshold

    # We check our calculate z-statistic against the calculated threshold and return a decision
    if t < -threshold or t > threshold:
        return "Reject"
    else:
        return "Do not reject"

# We undertake a hypothesis test to see if there is a difference between tipping rates when the temperature is below
# 20 and above 20
print "\n\nUndertaking hypothesis test. H_0 is that the mean tipping percentage when the temperature is below 20 C is " \
      "the same as when it's above. H_1 is that they are unequal."

# First we recalculate the bins, so there's only two bins: above 20 degrees and below 20 degrees
combined["temperature_bin"] = pd.cut(combined["T"], [-20, 20, 40]).apply(lambda bin: "%s - %s" % (bin.left, bin.right))

# We calculate the means, standard deviations and N for each bin
means = combined.groupby("temperature_bin").mean()["tip_percentage"]
stds = combined.groupby("temperature_bin").std()["tip_percentage"]
counts = combined.groupby("temperature_bin").count()["tip_percentage"]

# We use our previously defined hypothesis test function
print hypothesisTestTwoProportions_twoSided_t(means, counts, stds, 0.05)

# We undertake a hypothesis test to see if there is a difference between tipping rates when there is precipitation
# and when there is not
print "\n\nUndertaking hypothesis test. H_0 is that the mean tipping percentage when there is precipitation vs. when " \
      "there is not. H_1 is that they are unequal. "

# Again we recalculate the bins, so there's only two bins: 0 precipitation or > 0 precipitation
combined["precipitation_bin"] = pd.cut(combined["RRR"], [-1, 0, 50]).apply(
    lambda bin: "%s - %s" % (bin.left, bin.right))

# We calculate the means, standard deviations and N for each bin
means = combined.groupby("precipitation_bin").mean()["tip_percentage"]
stds = combined.groupby("precipitation_bin").std()["tip_percentage"]
counts = combined.groupby("precipitation_bin").count()["tip_percentage"]

# We use our previously defined hypothesis test function
print hypothesisTestTwoProportions_twoSided_t(means, counts, stds, 0.05)
