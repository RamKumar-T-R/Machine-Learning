import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 10 - Evaluating Regression Models Performance\data.csv')
X = dataset.iloc[:, 0:-1].values
Y = dataset.iloc[:, -1].values

# Splitting training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Using Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set
Y_pred = regressor.predict(X_test)  

# Evaluating the performace
from sklearn.metrics import r2_score
performance = r2_score(Y_test, Y_pred)
