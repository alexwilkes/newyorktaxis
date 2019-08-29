from __future__ import division
import urllib
import zipfile
import pandas as pd
import random
from utility import printHeader

################################################
# Configuration parameters - these should be completed prior to running the script
################################################

# Populate the main URL of the site
baseURL = "https://s3.amazonaws.com/nyc-tlc/trip+data/"

# Include the years of interest in the following list
yearsRequired = [2018]

# Include the months of interest as integers. range(1,13) will download all months.
monthsRequired = range(1,13)

# Include the files of interest. Possible values are 'yellow', 'green' and 'fhv.'
cabColors = ["yellow","green"]

# Define the local target directory for files to be downloaded to
targetDir = "data/"

# Define the proportion of each file we want to include in the dataset
sample_proportion = 1/120

# Define the path for the output file for tabular data
output = "data/combined_pre_cleaning.pickle"

# End of configuration parameters
################################################


###########################
# Spatial Data
###########################
printHeader("Spatial data")

# Download the file form the TLC website
urllib.urlretrieve("https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip","taxi_zones.zip")

# Unzip the file
print "Unzipping spatial data"
taxi_zones_zip = zipfile.ZipFile("taxi_zones.zip", 'r')
taxi_zones_zip.extractall("spatialref")
taxi_zones_zip.close()
print "Unzipping spatial data complete"

###########################
# Tabular Data
###########################
printHeader("Tabular Data")

# Create a list of the tabular data we intend to download
urls = [baseURL + cabColor + "_tripdata_" + str(year) + "-" + "%02d" % month + ".csv" for cabColor in cabColors for month in monthsRequired for year in yearsRequired]

# Next we download the tabular data
for url in urls:
    fname = url.split("/")[-1]
    print "Downloading tabular data: %s" % fname
    urllib.urlretrieve(url, targetDir + fname)

print "Tabular Data Downloads Complete."

# We create a list of the local files we have downloaded
localFiles = [targetDir + cabColor + "_tripdata_" + str(year) + "-" + "%02d" % month + ".csv" for cabColor in cabColors
              for month in monthsRequired for year in yearsRequired]

# We create a dictionary with the separate pandas dataframes in here. They're read in using the read_csv method which
#  takes the filename. We also tell pandas to skip some rows, and we provide a simple lambda function so we only get
# a sample of the population in the proportion specified.
data_dictionary = {localFile: pd.read_csv(localFile,skiprows=lambda i: i>0 and random.random() > sample_proportion)
                   for localFile in localFiles}

# For each dataframe we've loaded, we record the associated filename and colour of the cab.
for filename,df in data_dictionary.iteritems():
    df["filename"] = filename
    colour = filename.split("/")[1].split("_")[0]
    df["colour"] = colour

# We standardise the names of the features and drop two of the green-only features that we don't need
    df.rename(inplace=True,index=str,columns={"lpep_dropoff_datetime":"dropoff_datetime"})
    df.rename(inplace=True,index=str,columns={"lpep_pickup_datetime":"pickup_datetime"})
    df.rename(inplace=True,index=str,columns={"tpep_dropoff_datetime":"dropoff_datetime"})
    df.rename(inplace=True,index=str,columns={"tpep_pickup_datetime":"pickup_datetime"})
    if colour == "green": df.drop(inplace=True,columns=['ehail_fee', 'trip_type'])

# We then create a new dataframe combining all of them
combined = pd.concat(data_dictionary.values(),axis=0,sort=True)

# Finally we save the dataframe we created
combined.to_pickle(output)
print "Saved %s rows to %s" % (len(combined), output)
