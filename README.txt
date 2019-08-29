# Predicting Tips in the New York Taxi Market

# Contents
01_Data_Acquisition.py - Downloads spatial and tabular data from the NYC TLC website, samples it and stores it locally
02_Data_Cleaning_Tabular.py - Implements data cleaning on the tabular data
03_Data_Cleaning_Spatial.py - Implements data cleaning on the spatial data
04_Data_Cleaning_Weather.py - Implements data cleaning on the weather data
05_New_Feature_Creation.py - Implements new feature creation
06_Initial_Exploration.py - Produces a number of plots relating tipping to key features
07_Regression.py - Regression approach to tip prediction
08_Classification - Classification approach to tip prediction
utility.py - This contains a number of utility functions used by the scripts and must be included in the same location. It is not intended to be run directly.

# Instructions
Run all files except utility.py in order of the two digit prefix of the filename.

# Notes
- All scripts are written in Python 2.7.
- Scripts should be run in order as some have dependencies on data produced by the previous scripts.
- 01_Data_Acquisition.py can be configured to download data from specific months or years. It is recommended that at least one full year's data is downloaded.
- You will need to manually deploy the weather data CSV to a directory called data in the location the script runs from i.e. ./data/Weather_extract_2018.csv. The rest of the data will be downloaded and unzipped automatically by the scripts.
- You will also need to create the directory ./figures where the plots will be created.

# Dependencies
Beyond core python 2.7 libraries, you will need the following packages installed to run these scripts:
- numpy
- pandas
- geopandas
- keras
- SKLearn
- XGBoost
- scipy
- statsmodels
- prettytable
- matplotlib
- shapely
