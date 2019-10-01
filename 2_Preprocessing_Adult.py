'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: adult (available at "https://archive.ics.uci.edu/ml/index.php")
Objective: Presprcessing
'''
import numpy as np
import pandas as pd

#The given file doesn't has column names and missing values are depicted by ' ?'
#By studying index file, we give relevant column names and replace missing values by
#nan which is standard.
dataset = read_csv('Adress of csv file', names = ['age',
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
                                                  'salary'], na_values = ' ?'))

#To get insights regarding data
dataset.info()
dataset.describe()
dataset.isnull().sum()

#X will be info, y is what is to be predicted
X = dataset.iloc[:, 1:14].value
y = dataset.iloc[:, -1].value

#Since we found only columns- 1,6,13 to have missing values
#and all are string type
#We use Pandas fillna to fill in values
temp = pd.DataFrame(X[:, [1, 6, 13]])

#To check most frequent element
temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()

#Fill with most frequent element
temp[0] = temp[0].fillna(' Private')
temp[1] = temp[1].fillna(' Prof-specialty')
temp[2] = temp[2].fillna(' United-States')

X[;, [1, 6, 13]] = temp
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
y = lab.fit_transform(y)
lab.classes_                                        #Run to see classifications in y

#Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#X is ready!
