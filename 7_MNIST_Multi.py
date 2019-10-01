#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: MNIST (Available in sklearn)

Decision Tree Classifier, K-Neighbors Classifier and Logistic regression applied on MNIST dataset.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata('MNIST original')

#Alternative 2
dataset = pd.read_csv('mnist_train.csv')

#Alternative 3
from sklearn.datasets import load_digits
dataset_1 = load_digits()

#X = dataset_1.data
#y = dataset_1.target

#If 2nd alternative was taken
dataset.describe()
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


some_digit = X[3007]
#some_digit_image = some_digit.reshape(8, 8)
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 15)

#If fetch_mldata worked
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

#If 2nd alternative was taken
dataset_test = pd.read_csv('mnist_test.csv')
X_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

knn.fit(X_train, y_train)
knn.score(X_test, y_test)

dtc.fit(X, y)
dtc.score(X_test, y_test)

dtc.predict(X[[2, 935, 13948, 27639, 40228, 57230], 0:784])
