'''
Author: Ashish Ranjan
Lang: Python3.7
IDE: Spyder 3.3 (Anaconda3)
Data: adult (available at "https://archive.ics.uci.edu/ml/index.php")
      housing (available at "https://archive.ics.uci.edu/ml/index.php")
Objective: Applying Linear Regression on adult dataset
           Applying Logistic regression on housing dataset
'''
import numpy as np
import pandas as pd

##### Adult Dataset #####

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

#Split data in train-test categories.
#In my case, best results were acheived in case of 85%-15% split.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15)

#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#Check performance
log_reg.score(X_test, y_test)

##### Housing Dataset #####

dataset = pd.read_csv('Adress of csv file')

#To get insights regarding data
dataset.info()
dataset.describe()
dataset.isnull().sum()

#8th column contain values to be predicted.
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]].values
y = dataset.iloc[:, 8].values

#4th column needs to be imputed
from sklearn.preprocessing import Imputer
imp = Imputer(strategy = 'median')
X[:, [4]] = imp.fit_transform(X[:, [4]])

#Last column needs Label Encoding
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, -1] = lab.fit_transform(X[:, -1])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features=[-1])
X = one.fit_transform(X)

X = X.toarray()

#Turns out Data was already scaled, but still i scaled it here.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Split data in train-test.
#72%-25% split yeilded best result in my case.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

#Apply Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

#Check performance
lin_reg.score(X_test, y_test)
