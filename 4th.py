'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: Randomly generated
Objective: Apply Polynomial Regression
'''

import numpy as np
import matplotlib.pyplot as plt

#Generate any curve and random set of data arround it
m = 100
X = 6*np.random.randn(m, 1) - 3
y = X**2 + 2*X + 2 + np.random.randn(m, 1)

#Plotting a small span to see random distribution of data
plt.scatter(X, y)
plt.axis([-5, 5, 0, 10])
plt.show()

#Preprocess data to make it suitable to pass into linear regressor
#As polynomial regression is special case for linear
#where features are related to each other
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly.fit_transform(X)

#Apply Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

#Generate evenly spaced data to use for predictions
X_new = np.linspace(-5, 5, 100).reshape(-1, 1)
X_new_poly = poly.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)

#Plot and see if the trend shown is correct
plt.scatter(X, y)
plt.plot(X_new, y_new, c = 'r')
plt.axis([-5, 5, 0, 10])
plt.show()

#See the predicted curve and confirm weather
#it is close to one used to generate random data
lin_reg.coef_
lin_reg.intercept_
