#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: Salary (UCI Repository)

KNeighborsClassifier from sklearnis used. After analysis, k = 4 is found to be optimal here.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)
knn.score(X, y)




dataset = pd.read_csv('sal.csv', names = ['age',
                                   'workclass',
                                   'fnlwgt',
                                   'education',
                                   'education-num',
                                   'marital-status',
                                   'occupation',
                                   'relationship',
                                   'race',
                                   'gender',
                                   'capital-gain',
                                   'capital-loss',
                                   'hours-per-week',
                                   'native-country',
                                   'salary'], na_values = ' ?')

X = dataset.iloc[:, :14].values
y = dataset.iloc[:, -1].values

dataset.isnull().sum()

temp = pd.DataFrame(X[:,[1, 6, 13]])

#To check most frequent element
temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()

#Fill with most frequent element
temp[0] = temp[0].fillna(' Private')
temp[1] = temp[1].fillna(' Prof-specialty')
temp[2] = temp[2].fillna(' United-States')

X[:, [1, 6, 13]] = temp
del(temp)

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

#If data is interpreted as float in following lines(bug in some versions)
#and error is displayed, use X[:, 1].astype(str) instead and so on
X[:, 1] = lab.fit_transform(X[:, 1])                #Encoding Work Class
X[:, 3] = lab.fit_transform(X[:, 3])                #Encoding Education Class
X[:, 5] = lab.fit_transform(X[:, 5])                #Encoding Marital-status Class
X[:, 6] = lab.fit_transform(X[:, 6])                #Encoding Occupation Class
X[:, 7] = lab.fit_transform(X[:, 7])                #Encoding Relationship Class
X[:, 8] = lab.fit_transform(X[:, 8])                #Encoding Race Class
X[:, 9] = lab.fit_transform(X[:, 9])                #Encoding Gender Class
X[:, 13] = lab.fit_transform(X[:, 13])              #Encoding Native-country Class

#Dummy Variable Encoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1, 3, 5, 6, 7, 8, 9, 13])
X = one.fit_transform(X)

X = X.toarray()

y = lab.fit_transform(y)
lab.classes_                                        #Run to see classifications in y

#Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(X_train, y_train)
knn.score(X_test, y_test)
