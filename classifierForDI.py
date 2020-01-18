import numpy as np
import matplotlib.pyplot as plt
import PyML

classifier = SVM()

classifier.train(trainData)
r = classifier.test(testData)
ber = 1- r.getBalancedSuccessRate()
epsilon = ber
confusionMatrix = r.getConfusionMatrix()

beta = confusionMatrix[0,1]/(confusionMatrix[0,0] + confusionMatrix[0,1])
epsilonPrime = 1/2 - beta/8 #allowable BER threshold

#We want to check whether epsilonPrime < epsilon. If so, then no DI







# from sklearn import datasets
# from sklearn.ensemble import AdaBoostClassifier
#
# # n_estimators = 400 #TODO: Consider n_estimators later?
#
# X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
#
# X_test, y_test = X[2000:], y[2000:]
# X_train, y_train = X[:2000], y[:2000]
#
# ada_real = AdaBoostClassifier(
#     algorithm="SAMME.R")
# print(ada_real.fit(X_train, y_train))
