# Standard Data Wrangling Libraries
from __future__ import division
import pandas as pd
import numpy as np

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# StatsModels
import statsmodels.api as sm

# Evaluation
from sklearn.metrics import r2_score

# Visualisation
import matplotlib.pyplot as plt

# Utilities
from prettytable import PrettyTable
from utility import printHeader

# Keras by default is extremely verbose about the processes/threads it uses so we switch this off.
import os
os.environ['KMP_WARNINGS'] = 'off'

# Our plots often refer to yellow/green, I have chosen hex-codes which match the official colours of NYC taxis.
colorscheme = ["#8db600", "#fce300"]

# We load our cleaned tabular data file from the pickle file we saved.
combined = pd.read_pickle("data/combined_post_new_features.pickle").dropna()

# We drop a number of collinear features. They are collinear because they were created by one-hot encoding.
collinear_features = ["dropoff_borough_EWR", "pickup_borough_EWR", "colour_1", "temporal_dayofweek_6",
                      "temporal_month_12"]
combined = combined.drop(columns=collinear_features)

# We also rescale the coordinates to make them more interpretable
combined["pickup_x"] = (combined["pickup_x"] - combined["pickup_x"].mean()) / 1000
combined["pickup_y"] = (combined["pickup_y"] - combined["pickup_y"].mean()) / 1000
combined["dropoff_x"] = (combined["dropoff_x"] - combined["dropoff_x"].mean()) / 1000
combined["dropoff_y"] = (combined["dropoff_y"] - combined["dropoff_y"].mean()) / 1000

###########################
# Features
###########################

# We create explicit lists of each category of feature we are interested in
features_general = ["pickup_x", "pickup_y", "dropoff_x", "dropoff_y", "passenger_count", "trip_distance", "duration"]
features_dropoff = [feature for feature in combined.columns if feature[0:len("dropoff_borough_")] == "dropoff_borough_"]
features_pickup = [feature for feature in combined.columns if feature[0:len("dropoff_pickup_")] == "pickup_borough_"]
features_ratecode = [feature for feature in combined.columns if feature[0:len("Ratecode_")] == "Ratecode_"]
features_colour = [feature for feature in combined.columns if feature[0:len("colour_")] == "colour_"]
features_temporal = [feature for feature in combined.columns if feature[0:len("temporal_")] == "temporal_"]
features_weather = ["T", "RRR"]

# We create our big list of features and print it out
printHeader("Features")
features = features_general + features_dropoff + features_pickup + \
           features_colour + features_temporal + features_weather
for feature in features: print "* %s" % feature

###########################
# Train/Test Split
###########################
printHeader("Train/Test Split")
X_train, X_test, y_train, y_test = train_test_split(combined[features], combined["tip_percentage"], test_size=0.33)

print "X_train: %s" % len(X_train)
print "X_test: %s" % len(X_test)
print "y_train: %s" % len(y_train)
print "y_test: %s" % len(y_test)

###########################
# Linear Regression
###########################
printHeader("Linear Regression")

# We remove features from the regression until all are significant. We initialise this list for recording what gets
# removed.
notSignificantFeatures = []

# We train the initial Linear Regression Model
result = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# We use a while loop to remove non-significant features from the linear regression until all features are
# significant (at 95% confidence)
continueIterating = True
while continueIterating:
    notSignificantFeatures = notSignificantFeatures + [feature for feature, p in result.pvalues.iteritems() if
                                                       p >= 0.05]

    # If the constant is found to be not significant, we don't want to remove it so we fix this
    if "const" in notSignificantFeatures: notSignificantFeatures.remove("const")

    # Retrain the regression model
    result = sm.OLS(y_train, sm.add_constant(X_train.drop(columns=notSignificantFeatures))).fit()

    # Determine if we need to continue iterating
    continueIterating = True if sum(result.pvalues >= 0.05) > 0 else False

# We print out the features were removed
print "The following features were removed:"
for feature in notSignificantFeatures: print "- %s" % feature

# Finally we print out the weights, loading them in to a pandas dataframe to make printing easy
print pd.concat([
    pd.Series(result.params, name="Weights").apply(lambda weight: "{:.2f}".format(weight)),
    pd.Series(result.pvalues < 0.05, name="Significant") # Returns True if the p-value is less than 0.05
], axis=1)

###########################
# Plotting Fig 24 - Distribution of predictions vs. actuals for Linear Regression
###########################
printHeader("Plotting Fig 24 - Distribution of predictions vs. actuals for Linear Regression")

# Calculate predictions on the test data using the model so we can evaluate it
predictions = result.predict(sm.add_constant(X_test.drop(columns=notSignificantFeatures)))

# Plot the predictions vs. the actuals
base = pd.Series(predictions).plot(kind="kde", label="Prediction")
y_test.plot(ax=base, kind="kde", label="Actual")

# Tidy up the axes
plt.legend()
plt.xlabel("Tip Percentage")
plt.xlim([-5, 30])

# Set the tight layout flag and save the figure
plt.tight_layout()
plt.savefig("figures/fig24.png")
print "Saved fig24.png.\n"
plt.clf()

# Print out the r^2 score
print "r2 score: %.3f" % r2_score(y_test, predictions)

###########################
# Neural Network
###########################
printHeader("Neural Network")

# We have to scale the data to the (0,1) interval to train the neural network effectively.
min_max_scaler = MinMaxScaler()
X_scale = min_max_scaler.fit_transform(combined[features])

# Because we rescaled, we need to re-run train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scale, combined["tip_percentage"], test_size=0.33)

# We set up the structure of the neural network
model = keras.Sequential()
model.add(layers.Dense(64, input_dim=len(features), activation='relu', use_bias=True))
model.add(layers.Dense(32, activation='relu', use_bias=True))
model.add(layers.Dense(16, activation='relu', use_bias=True))
model.add(layers.Dense(8, use_bias=True))
model.add(layers.Dense(4, use_bias=True))
model.add(layers.Dense(1, use_bias=True))

# We define the optimizer and compile the model
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# We fit the model
model.fit(X_train, y_train, validation_split=0.1, epochs=5)

###########################
# Plotting Fig 25 - Distribution of predictions vs. actuals for Neural Network
###########################
printHeader("Plotting Fig 25 - Distribution of predictions vs. actuals for Neural Network")

# To evaluate the model, we run the prediction method on the test data
predictions = model.predict(X_test)

# We plot the predictions vs the actuals
base = pd.Series(predictions[:, 0]).plot(kind="kde", label="Prediction")
y_test.plot(ax=base, kind="kde", label="Actual")

# Tidy up the axes
plt.legend()
plt.xlabel("Tip Percentage")
plt.title("Distribution of predictions vs. actuals for Neural Network Regression")
plt.xlim([-5, 30])

# Set the tight layout flag and save the figure
plt.tight_layout()
plt.savefig("figures/fig25.png")
print "Saved fig25.png.\n"
plt.clf()

# We calculate r^2 for the test data
print "r2 score: %.3f" % r2_score(y_test, predictions)

###########################
# Random Forests Regression
###########################
printHeader("Random Forests Regression")

# We create the regression using SKLearn's regresssor and fit it
regr = RandomForestRegressor(max_depth=15, n_estimators=50)
regr.fit(X_train, y_train)
print "Trained"


###########################
# Plotting Fig 26 - Distribution of predictions vs. actuals for Random Forests
###########################
printHeader("Plotting Fig 26 - Distribution of predictions vs. actuals for Random Forests")

# To evaluate the model we predict over the test data
predictions = regr.predict(X_test)

# We plot the predictions vs the actuals
base = pd.Series(predictions).plot(kind="kde", label="Prediction")
y_test.plot(ax=base, kind="kde", label="Actual")

# Tidy up the axes
plt.legend()
plt.xlabel("Tip Percentage")
plt.title("Distribution of predictions vs. actuals for RF Regression")
plt.xlim([-5, 30])

# Set the tight layout flag and save the figure
plt.tight_layout()
plt.savefig("figures/fig26.png")
print "Saved fig26.png.\n"
plt.clf()

# We calculate r^2 for the test data
print "r2 score: %.3f" % r2_score(y_test, predictions)

###########################
# Plotting Fig 27 - Feature Importances for Random Forests
###########################
printHeader("Plotting Fig 27 - Feature Importances for Random Forests")

# We load the importances calculated by SKLearn in to a pandas dataframe and use the pandas plot function
pd.DataFrame({"Feature Importance": regr.feature_importances_}, index=features) \
    .sort_values(by="Feature Importance", ascending=False) \
    .plot(kind="bar", figsize=(12, 5), legend=False, color="black")

# Set the tight layout flag and save the figure
plt.tight_layout()
plt.savefig("figures/fig27.png")
print "Saved fig27.png.\n"
plt.clf()