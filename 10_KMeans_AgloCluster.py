#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: Randomly generated.

K-Means applied on data generated in groups to guess the number of groups.
Same process is repeated with Agglomerative Clustering.

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 1000, centers = 5, cluster_std = 0.7)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans

wcv = []

for i in range(1, 16):
    km = KMeans(n_clusters = i)
    km.fit(x)
    wcv.append(km.inertia_)

plt.plot(range(1, 16), wcv)

km = KMeans(n_clusters = 5)
y_pred = km.fit_predict(x)

plt.scatter(x[y_pred==0, 0], x[y_pred==0, 1], c = 'r')
plt.scatter(x[y_pred==1, 0], x[y_pred==1, 1], c = 'b')
plt.scatter(x[y_pred==2, 0], x[y_pred==2, 1], c = 'g')
plt.scatter(x[y_pred==3, 0], x[y_pred==3, 1], c = 'y')

#########################################################

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 6, cluster_std = .7)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(x, method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters = 5)
y_pred = hca.fit_predict(x)

for i in range(5):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1])
plt.show()
