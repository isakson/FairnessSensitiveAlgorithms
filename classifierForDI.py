# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from PyML import *
# from DataSet import DataSet
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier

ds = DataSet()
ds.loadData("ClassifiedRicciData.csv", ["Race"], "Class")
df = ds.DataFrame

train = df.sample(.7)
train_test = train_test_split(df, train_size=.8, test_size=.2, shuffle=True)
print(train_test)

classifier = SVC(class_weight='balanced')
classifier.fit(train_test[0])
classifier.predict(train_test[1])


'''
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
