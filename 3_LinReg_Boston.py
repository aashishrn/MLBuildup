'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: load_boston (available with sklearn)
Objective: Apply Linear Regression
'''

from sklearn.datasets import load_boston
dataset = load_boston()

X = dataset.data
y = dataset.target

# Applying linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X, y)

# Checking Prediction Acurracy on dataset
lin_reg.score(X, y)

# Result- 0.74
