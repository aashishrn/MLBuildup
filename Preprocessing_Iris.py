'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: iris (available with sklearn library)

As a part of data preprocessing, we'll plot all 4 data columns available in database
and observe wcich combination is ideal for classification of data.
We have 3 objects, given value- 0,1,2
We'll for which combination, making distinction is easiest.
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

# Sepal length vs width
plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Setosa')
plt.scatter(X[y==1,0], X[y==1,1], c='g', label='Versicolor')
plt.scatter(X[y==2,0], X[y==2,1], c='b', label='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.title('Analysis on iris Dataset')
plt.show()

# Petal length vs width
plt.scatter(X[y==0,2], X[y==0,3], c='r', label='Setosa')
plt.scatter(X[y==1,2], X[y==1,3], c='g', label='Versicolor')
plt.scatter(X[y==2,2], X[y==2,3], c='b', label='Virginica')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.title('Analysis on iris Dataset')
plt.show()

# Sepal length vs Petal width
plt.scatter(X[y==0,0], X[y==0,3], c='r', label='Setosa')
plt.scatter(X[y==1,0], X[y==1,3], c='g', label='Versicolor')
plt.scatter(X[y==2,0], X[y==2,3], c='b', label='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.title('Analysis on iris Dataset')
plt.show()

# Petal length vs Sepal width
plt.scatter(X[y==0,2], X[y==0,1], c='r', label='Setosa')
plt.scatter(X[y==1,2], X[y==1,1], c='g', label='Versicolor')
plt.scatter(X[y==2,2], X[y==2,1], c='b', label='Virginica')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.title('Analysis on iris Dataset')
plt.show()

# Sepal vs Petal length
plt.scatter(X[y==0,0], X[y==0,2], c='r', label='Setosa')
plt.scatter(X[y==1,0], X[y==1,2], c='g', label='Versicolor')
plt.scatter(X[y==2,0], X[y==2,2], c='b', label='Virginica')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.title('Analysis on iris Dataset')
plt.show()

# Sepal vs Petal width
plt.scatter(X[y==0,1], X[y==0,3], c='r', label='Setosa')
plt.scatter(X[y==1,1], X[y==1,3], c='g', label='Versicolor')
plt.scatter(X[y==2,1], X[y==2,3], c='b', label='Virginica')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()
plt.title('Analysis on iris Dataset')
plt.show()
