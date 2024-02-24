import pandas as pd
import numpy as np
import matplotlib.pyplot as mlp

# Importing datasets
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression\\50_Startups.csv')
X = dataset.iloc[:, 0: -1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1: 6]

# Splitting trainset and testset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Using linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Actually this is the machine learing model
regressor.fit(X_train, Y_train)

# Predicting the result
Y_pred = regressor.predict(X_test)
np.printoptions(precesion = 2)
result = np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), axis = 1) 