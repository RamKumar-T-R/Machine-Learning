import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 9 - Random Forest Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500)
regressor.fit(X, Y)

# Predict the Y value of 6.5
result = regressor.predict([[6.5]])

# Visualizing
plt.title('Positions vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.scatter(X, Y)
plt.plot(X, regressor.predict(X))