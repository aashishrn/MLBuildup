#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: Bank (available on UCI repository)

Decision Tree Classifier is applied after preprocessing Bank data.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bank-full.csv', sep = ";")#, na_values = 'unknown')

dataset.info()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

temp =

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

X[:, 1] = lab.fit_transform(X[:, 1])
X[:, 2] = lab.fit_transform(X[:, 2])
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 4] = lab.fit_transform(X[:, 4])
X[:, 6] = lab.fit_transform(X[:, 6])
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 10] = lab.fit_transform(X[:, 10])
X[:, 15] = lab.fit_transform(X[:, 15])
y = lab.fit_transform(y)

test = pd.DataFrame(X)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1, 2, 3, 4, 6, 7, 8, 10, 15])
X = one.fit_transform(X)

X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 15)

dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)


from sklearn.tree import export_graphviz

export_graphviz(dtc, out_file = 'tree.dot')

import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
