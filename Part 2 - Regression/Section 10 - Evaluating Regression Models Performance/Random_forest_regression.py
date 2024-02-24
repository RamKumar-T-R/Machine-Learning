import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 10 - Evaluating Regression Models Performance\data.csv')
X = dataset.iloc[:, 0: -1].values
Y = dataset.iloc[:, -1].values

# Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Using Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500)
regressor.fit(X_train, Y_train)

# Predicting the result
Y_pred = regressor.predict(X_test)

# Evaluating the performance
from sklearn.metrics import r2_score
performance = r2_score(Y_test, Y_pred)