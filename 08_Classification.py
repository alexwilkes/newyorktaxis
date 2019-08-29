# Standard Data Wrangling Libraries
from __future__ import division
import pandas as pd
import numpy as np
import math

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# # TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# StatsModels
import statsmodels.api as sm

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import export_graphviz

# Visualisation
import matplotlib.pyplot as plt

# Utilties
from utility import printHeader
from utility import printPrettyCM

# Keras by default is extremely verbose about the processes/threads it uses so we switch this off.
import os
os.environ['KMP_WARNINGS'] = 'off'

# Our plots often refer to yellow/green, I have chosen hex-codes which match the official colours of NYC taxis.
colorscheme = ["#8db600", "#fce300"]

###########################
# Loading Data
###########################
printHeader("Loading Data")
# We load our cleaned tabular data file from the pickle file we saved.
# To speed up training, you can limit the sample size of this by sampling a percentage
combined = pd.read_pickle("data/combined_post_new_features.pickle")
print "%s records loaded from dataset." % len(combined)

###########################
# Features
###########################
printHeader("Features")

# To help manage all the new features we've created, we construct a list of all the features we want to include in
# our model
features_general = ["pickup_x", "pickup_y", "dropoff_x", "dropoff_y", "passenger_count", "trip_distance", "duration"]
features_dropoff = [feature for feature in combined.columns if feature[0:len("dropoff_borough_")] == "dropoff_borough_"]
features_pickup = [feature for feature in combined.columns if feature[0:len("dropoff_pickup_")] == "pickup_borough_"]
features_ratecode = [feature for feature in combined.columns if feature[0:len("Ratecode_")] == "Ratecode_"]
features_colour = [feature for feature in combined.columns if feature[0:len("colour_")] == "colour_"]
features_temporal = [feature for feature in combined.columns if feature[0:len("temporal_")] == "temporal_"]
features_weather = ["T", "RRR"]

# Because of our use of dummy variables, a number of our features are collinear.
# Some models cannot cope with this so we keep a list of them so we can remove them appropriately.
collinear_features = ["dropoff_borough_Staten Island", "pickup_borough_Staten Island", "colour_1",
                      "temporal_dayofweek_6", "temporal_month_12"]
features = features_general + features_dropoff + features_pickup + features_colour + features_temporal + features_weather
features = [feature for feature in features if feature not in collinear_features]

# We print a list of features
for feature in features: print "* %s" % feature

# We also name the classes
classnames = ["Non-Zero", "Zero"]

###########################
# Train/Test Split
###########################
printHeader("Train/Test Split")

# We split the dataset in to training and test data
combined["Train_Test"] = np.random.choice(["Train", "Train", "Test"], combined.shape[0])
train_data = combined[combined["Train_Test"] == "Train"]
test_data = combined[combined["Train_Test"] == "Test"]

###########################
# Balancing the data
###########################
printHeader("Balancing the data")
sizeOfMinorityClass = min(sum(train_data["zero_tip"] == 0), sum(train_data["zero_tip"] == 1))
print "\n\nThe smaller class size is: %s\n\n" % sizeOfMinorityClass

balanced_train_data = pd.concat([
    train_data[train_data["zero_tip"] == 0].sample(sizeOfMinorityClass),
    train_data[train_data["zero_tip"] == 1].sample(sizeOfMinorityClass)]) \
    .sample(frac=1)  # This 'sampling' is an efficient way to shuffle the dataset prior to training.

###########################
# Fitting and Evaluating non-neural network classifiers
###########################
printHeader("Fitting and Evaluating non-neural network classifiers")

# We initialise a list to store the metrics in
classifier_metrics = []

# This function is used to evaluate a classifier. It takes the classifier, the predictions and class probabilities
# for the test data
def evaluateClassifier(clf_name, predictions, predictedProbabilities):
    print clf_name
    print "--------"

    # Print the confusion matrix
    printPrettyCM(confusion_matrix(test_data["zero_tip"], predictions), classnames)

    # print the key metrics
    print "Accuracy: %s" % '{0:.2f}'.format(accuracy_score(test_data["zero_tip"], predictions))
    print "Precision: %s" % '{0:.2f}'.format(precision_score(test_data["zero_tip"], predictions))
    print "Recall: %s" % '{0:.2f}'.format(recall_score(test_data["zero_tip"], predictions))
    print "F1 Score: %s\n\n\n" % '{0:.2f}'.format(f1_score(test_data["zero_tip"], predictions))

    # We want to build a RoC plot later so we store the required objects to do this
    fpr, tpr, thresholds = roc_curve(test_data["zero_tip"], predictedProbabilities[:, 1])
    RoC_curves[clf_name] = [fpr, tpr]

    # We append the relevant metrics to our list for presentation after everything is trained
    metrics = [
        clf_name,
        accuracy_score(test_data["zero_tip"], predictions),
        precision_score(test_data["zero_tip"], predictions),
        recall_score(test_data["zero_tip"], predictions),
        f1_score(test_data["zero_tip"], predictions),
        auc(fpr, tpr)
    ]
    classifier_metrics.append(metrics)

# The following dictionary is populated with the classifiers that we want to use
clf_dict = {
    "Naive Bayes": BernoulliNB(),
    "Logistic Regression": LogisticRegression(solver='lbfgs', multi_class='ovr'),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion="gini", min_samples_leaf=500),
    "Random Forests": RandomForestClassifier(n_estimators=100, max_depth=5),
    "XGBoost": XGBClassifier(),
}

# This function will export a graphviz plot of the tree classifier passed to it
def analyseTree(clf):
    # We plot the tree
    export_graphviz(clf, out_file="figures/fig29.dot")
    print "Figure 29 saved to figures/fig29.dot. This must be converted using DOT. Install graphviz and run the " \
          "following command: dot -Tpng figures/fig29.dot -o figures/fig29.png "

# This function will save a plot of the feature importances of a tree based classifier passed to it
def plotFeatureImportances(name, clf):
    # We take the feature importances calculated by SKLearn and load them in to a pandas dataframe so we can use the
    # appropriate plot function
    pd.DataFrame({"Feature Importance": clf.feature_importances_}, index=features) \
        .sort_values(by="Feature Importance", ascending=False) \
        .plot(kind="bar", figsize=(12, 5), legend=False, color="black")

    # Tidy up the plot
    plt.title("Feature Importances for classifier: %s" % name)
    plt.ylabel("Importance")
    plt.xlabel("Feature")

    # We change the filename depending on the classifier.
    if name == "Decision Tree":
        filename = "fig30.png"
    elif name == "Random Forests":
        filename = "fig31.png"

    # Set the tight layout flag and save the figure
    plt.tight_layout()
    plt.savefig("figures/%s" % filename)
    print "Saved %s.\n" % filename
    plt.clf()

# We create an empty dictionary to store the RoC Curves as we evalulate our classifiers
RoC_curves = {}

# We iterate through the classifiers, training them and evaluating them.
for name, clf in clf_dict.iteritems():
    # Train the classifier
    clf.fit(balanced_train_data[features], balanced_train_data["zero_tip"])

    # Calculate predictions and class probabilities so we can evaluate the classifier
    predictions = clf.predict(test_data[features])
    predictedProbabilities = clf.predict_proba(test_data[features])

    # Run our previously defined classifier evaluation function
    evaluateClassifier(name, predictions, predictedProbabilities)

    # In the special case of the decision tree, we plot the tree and feature importances
    if name == "Decision Tree":
        analyseTree(clf)
        plotFeatureImportances(name, clf)

    # In the special case of random forests, we plot the feature importances
    elif name == "Random Forests":
        plotFeatureImportances(name, clf)

###########################
# Reviewing logit model in more detail
###########################
printHeader("Reviewing logit model in more detail")

# We remove features that are not significant from the regression. A list is initialised to track those we remove
notSignificantFeatures = []

# The logistic regression is trained using the stats models package.
logit = sm.Logit(balanced_train_data["zero_tip"],
                 sm.add_constant(balanced_train_data[features].drop(columns=notSignificantFeatures)))
# The fit_regularized method is used to implement ridge regression, which is the default of SKLearn and we want to
# mirror this. We set a high number of maximum iterations as the model can take longer than the default to converge.
result = logit.fit_regularized(method='l1', maxiter=1000)

# We iterate the regression until all features are significant
continueIterating = True
while continueIterating:
    # Update the list of non-significant features
    notSignificantFeatures = notSignificantFeatures + [feature for feature, p in result.pvalues.iteritems() if
                                                       p >= 0.05]

    # If const is not significant, we don't want to remove it so we add it back in
    if "const" in notSignificantFeatures: notSignificantFeatures.remove("const")

    # We retrain the regression on our reduced feature set
    logit = sm.Logit(balanced_train_data["zero_tip"],
                     sm.add_constant(balanced_train_data[features].drop(columns=notSignificantFeatures)))
    result = logit.fit_regularized(maxiter=1000)

    # We continue iterating if the regression still has insignificant features
    if sum(result.pvalues >= 0.05) == 0: continueIterating = False

# We print out the features were removed
print "\nThe following features were removed as they were not significant:"
for feature in notSignificantFeatures: print "* %s" % feature

# We summarise the logistic regression model by loading the weights in to a pandas dataframe and printing it
logitSummary = pd.DataFrame(result.params, columns=["Weight"])
logitSummary["exp(weight)"] = logitSummary["Weight"].apply(math.exp).apply(lambda x: "%.2f" % (x))
logitSummary["Individually Significant"] = (result.pvalues < 0.05)
print "\nWeights for Logistic Regression"
print logitSummary

###########################
# Neural Network
###########################
printHeader("Neural Network")

# We need to scale all the features to (0,1) to for effective training of the network
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(balanced_train_data[features])

# We build the structure of the neural network
model = keras.Sequential([
    layers.Dense(64, activation="sigmoid", input_shape=[len(features)]),
    layers.Dense(32, activation="sigmoid"),
    layers.Dense(16, activation="sigmoid"),
    layers.Dense(2, activation="sigmoid")
])

# Define our optimizer and compile our model
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# We train the network
model.fit(X_scaled, to_categorical(balanced_train_data["zero_tip"]), validation_split=0.1, epochs=5)

# We calculate class probabilities using the trained network. We convert these to hard predictions by selecting the
# most likely class.
predictionProbabilities = model.predict_proba(min_max_scaler.transform(test_data[features]))
predictions = [1 if p[1] > p[0] else 0 for p in predictionProbabilities]

# We run our classifier evaluation function on the neural network
evaluateClassifier("Neural Net", predictions, predictionProbabilities)

###########################
# Summary Metrics
###########################
printHeader("Summary Metrics")

# We load the metrics we've been storing in to a pandas dataframe and print them out, rounding them to 2dp
print pd.DataFrame(classifier_metrics,
                   columns=["Name", "Accuracy", "Precision", "Recall", "F1 Score", "Area Under Curve"]) \
    .set_index("Name") \
    .round(2)

###########################
# Figure 28 - RoC Curves
###########################
printHeader("Figure 28 - RoC curves")

# Create a figure to plot on.
plt.figure(figsize=(10, 10))

# Set the axes labels and range
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim((0, 1))
plt.ylim((0, 1))

# Plot a straight line through the origin
plt.plot((0, 1), (0, 1), color="gray", linestyle="--")

# Iterate through each item in the dictionary, calculating AuC and plotting the RoC curve
for name, curve in RoC_curves.iteritems():
    areaUnderCurve = auc(curve[0], curve[1])
    plt.plot(curve[0], curve[1], label='%s. Area Under Curve: %0.2f)' % (name, areaUnderCurve))

# Add legend to the lower right hand side
plt.legend(loc="lower right")

# We initialise a list to store the metrics in
classifier_metrics = []
plt.savefig("figures/fig28.png")
print "Saved fig28.png.\n"
plt.clf()
