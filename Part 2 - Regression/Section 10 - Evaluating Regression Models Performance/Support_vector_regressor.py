import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 10 - Evaluating Regression Models Performance\data.csv')
X = dataset.iloc[:, 0:-1].values
Y = dataset.iloc[:, -1].values

Y = Y.reshape(len(Y), 1)

# Splitting into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.fit_transform(Y_test)

# Using Support Vector Regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, Y_train)

# Predicting the result
Y_pred = sc_Y.inverse_transform(regressor.predict(X_test).reshape(len(X_test), 1))

# Compatison
compare = np.concatenate((Y_pred, sc_Y.inverse_transform(Y_test))) # You should give the double paranthesis sice Y_pred and Y_test should be in same parameter

# Evaluating the performance
from sklearn.metrics import r2_score
performance = r2_score(sc_Y.inverse_transform(Y_test), Y_pred)

