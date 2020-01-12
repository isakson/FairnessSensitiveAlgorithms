import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier

# n_estimators = 400 #TODO: Consider n_estimators later?

X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

X_test, y_test = X[2000:], y[2000:]
X_train, y_train = X[:2000], y[:2000]

ada_real = AdaBoostClassifier(
    algorithm="SAMME.R")
print(ada_real.fit(X_train, y_train))
