#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: Bank (available on UCI repository)

Sentiment Analysis of tweets for using NLTK, GaussianNB using words used in tweets.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import re

dataset = pd.read_csv('train.csv')
dataset['tweet'][0]

processed_tweet = []

for i in range(31962):

    tweet = re.sub('@[\w]*', ' ', dataset['tweet'][i])
    tweet = re.sub('[^a-zA-Z#]', ' ', dataset['tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(token) for token in tweet if not token in stopwords.words('english') ]
    tweet = ' '.join(tweet)
    processed_tweet.append(tweet)
    if(not i%1000):
        print(i)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)

X = cv.fit_transform(processed_tweet)
X = X.toarray()
y = dataset['label'].values

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X, y)

n_b.score(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 15)

dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
knn.score(X_test, y_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

print(cv.get_feature_names())
