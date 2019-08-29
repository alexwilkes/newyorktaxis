from __future__ import division
import geopandas as gpd
from shapely.geometry import Polygon

from utility import printHeader

###########################
# Loading the data and print out the CRS and features
###########################
printHeader("Loading the data and print out the CRS and features")
# Load and join the spatial data
gdf = gpd.read_file("spatialref/taxi_zones.shp")

# Print out the coordinate reference system
print "Coordinate system: %s\n" % gdf.crs["init"]

# Print out the features of the spatial data
print "Features:"
for feature in gdf.columns:
    print "* %s" % feature

###########################
# Check whether we should use the OBJECT ID or the LocationID as our index
###########################
printHeader("Check whether we should use the OBJECT ID or the LocationID as our index")

print "The following entries exist in the dataframe where LocationID is not the same as OBJECTID"
print gdf[(gdf["OBJECTID"] != gdf["LocationID"])]

# Index by OBJECTID
gdf = gdf.set_index("OBJECTID")

# Check for missing values
printHeader("Checking for missing values")
print "Rows in the geodataframe with NaN values:"
print gdf[gdf.isna().any(axis=1)]

# Check for extreme values in the borough column
printHeader("Checking for extreme values in the borough column")
print "Counts for zones by borough:"
print gdf["borough"].value_counts()

# We want to check whether any of the locations in the spatial data are not in or near New York city.
printHeader("Checking locations of all zones are in or near NYC")
# We create a polygon (rectangle setting out bounds we would expect to see all our data within
# These bounds were identified by using https://epsg.io/map#srs=4326&x=0.000000&y=0.000000&z=1&layer=streets
bounds = Polygon([(900000, 280000), (1070000, 280000), (1070000, 110000), (900000, 110000)])

# For each zone in the geopandas dataframe, we classify it as within bounds and sum those that are within the bounds
# we defined
numberWithinBounds = sum(gdf["geometry"].apply(lambda geometry: geometry.within(bounds)))
# Print the result
print "\n\nThe location of all zones was checked. %s out of %s locations were found to be within bounds."\
      % (numberWithinBounds, len(gdf))

###########################
# Save the spatial data in a pickle file
###########################
printHeader("Save the spatial data")
gdf.to_pickle("data/spatial.pickle")
print "Spatial data saved to data/spatial.pickle"