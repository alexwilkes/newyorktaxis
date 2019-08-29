# -*- coding: utf-8 -*-

from prettytable import PrettyTable
# This file contains a number of utility functions used by the scripts and must be included in the same location

# We print a number of percentages. This is a simple function for formatting them nicely as strings
def asPercentage(f):
    return "{:.1%}".format(f)

# We print a number of headers. This is a simple function for giving them some presence in the output
def printHeader(header):
    print "\n"
    print "#" * 50
    print "# " + header
    print "#" * 50

# We print a lot of confusion matrices, and this function prints them in a way that is easier to read
def printPrettyCM(cm, labels):
    # Print the absolute numbers
    print "Confusion Matrix (absolute counts)"

    # Create a table with the labels as headers
    t = PrettyTable([" "] + labels)

    # Iterate through the rows adding them to the pretty table
    for x in range(len(labels)):
        t.add_row([labels[x]] + list(cm[x]))

    print(t)

    # Print as percentages
    cm = cm / cm.sum()

    print "Confusion Matrix (percentages)"

    # Create a table with the labels as headers
    t = PrettyTable([" "] + labels)

    # Iterate through the rows adding them to the pretty table
    for x in range(len(labels)):
        row = [labels[x]] + ["{:.1f}%".format(100 * cell) for cell in cm[x]]
        t.add_row(row)

    print(t)