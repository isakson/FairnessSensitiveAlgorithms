from DataSet import DataSet
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ds = DataSet()
ds.loadData("BinaryClassifiedRicciData.csv", ["Race"], "Class")
dataFrame = ds.dataFrame

'''
Splits data into training and test sets by shuffling the order of the rows and splitting into 
two DataFrames according to the ratios provided.
    dataFrame (DataFrame) - the dataFrame to split into training and test sets
    fractionTrain (float between 0 and 1) - the percentage of the data that should be in the training set
    fractionTest (float between 0 and 1) - the percentage of the data that should be in the training set
        Note: fractionTrain and fractionTest should sum to 1.
    classifierCol (string) - the header for the column that the classifier should be trying to classify
'''
#TODO: write return for comment above

def splitDataIntoTrainTest(dataFrame, fractionTrain, fractionTest, classifierCol):
    train_test = train_test_split(dataFrame, train_size=fractionTrain, test_size=fractionTest, shuffle=True)
    print(train_test)

    trainClassifications = train_test[0][classifierCol]
    testClassifications = train_test[1][classifierCol]

    return train_test, trainClassifications, testClassifications

'''
Runs an SVC SVM classifier on the specified training and test sets.
'''
def classify(train_test, trainClassifications, testClassifications):
    classifier = SVC(class_weight='balanced')
    classifier.fit(train_test[0], trainClassifications)
    return classifier.predict(train_test[1])

'''
Computes the balanced error rate (BER) for the classifier results.
'''
def computeBER(train_test, classifierCol, classifierResults):
    bsr = balanced_accuracy_score(train_test[1][classifierCol], classifierResults)
    ber = 1 - bsr
    return ber

'''
Computes the parameter Beta for the algorithm.
'''
def computeBeta(train_test, classifierCol, classifierResults):
    confusionMatrix = confusion_matrix(train_test[1][classifierCol], classifierResults)
    beta = confusionMatrix[0, 1] / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
    return beta

classifierCol = "Race"
data_split = splitDataIntoTrainTest(dataFrame, .8, .2, classifierCol)
classifications = classify(data_split[0], data_split[1], data_split[2])
ber = computeBER(data_split[0], classifierCol, classifications)
beta = computeBeta(data_split[0], classifierCol, classifications)
epsilonPrime = 1 / 2 - beta / 8  # allowable BER threshold
if epsilonPrime < ber:
    print("No disparate impact.")
else:
    print("Possible disparate impact.")



'''
#The lines of code below use PyML instead of sklearn

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from PyML import *

def main():
    # ds = DataSet()
    # ds.loadData("ClassifiedRicciData.csv", ["Race"], "Class")

    df = pd.DataFrame(np.random.randn(100, 2))

    # df = ds.dataFrame
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    classifier = SVM()
    print "completed SVM"

    classifier.train(train)
    r = classifier.test(test)
    ber = 1 - r.getBalancedSuccessRate()
    epsilon = ber

    confusionMatrix = r.getConfusionMatrix()

    beta = confusionMatrix[0, 1] / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
    epsilonPrime = 1 / 2 - beta / 8  # allowable BER threshold

    # We want to check whether epsilonPrime < epsilon. If so, then no DI


if __name__ == "__main__":
    main()
'''


# n_estimators = 400 #TODO: Consider n_estimators later?
#
# X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
#
# X_test, y_test = X[2000:], y[2000:]
# X_train, y_train = X[:2000], y[:2000]
#
# ada_real = AdaBoostClassifier(
#     algorithm="SAMME.R")
# print(ada_real.fit(X_train, y_train))
