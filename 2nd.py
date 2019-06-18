import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/home/ashish/Downloads/adult.csv', header = None, index_col = False)
dataset.replace(" ?",np.nan)

X = dataset.iloc[:, :14].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer

sim_mean = SimpleImputer(strategy = "mean")
sim_median = SimpleImputer(strategy = "median")
sim_mode = SimpleImputer(strategy = "most_frequent")

X[:, 0] = sim_mean.fit_transform(X[:, 0].reshape(-1,1)).reshape(1,-1)[0]
X[:, 1] = sim_mode.fit_transform(X[:, 1].reshape(-1,1)).reshape(1,-1)[0]
X[:, 2] = sim_median.fit_transform(X[:, 2].reshape(-1,1)).reshape(1,-1)[0]
X[:, 3] = sim_mode.fit_transform(X[:, 3].reshape(-1,1)).reshape(1,-1)[0]
X[:, 4] = sim_mean.fit_transform(X[:, 4].reshape(-1,1)).reshape(1,-1)[0]
X[:, 5:10] = sim_mode.fit_transform(X[:, 5:10])
X[:, 10:12] = sim_mode.fit_transform(X[:, 10:12])
X[:, 12] = sim_median.fit_transform(X[:, 12].reshape(-1,1)).reshape(1,-1)[0]
X[:, 13] = sim_mode.fit_transform(X[:, 13].reshape(-1,1)).reshape(1,-1)[0]

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
for i in [1,3,5,6,7,8,9,13]:
    X[:, i] = lab.fit_transform(X[:, i])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])

X = one.fit_transform(X)

X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
