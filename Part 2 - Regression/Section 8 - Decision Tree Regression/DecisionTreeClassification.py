import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 8 - Decision Tree Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values

# Using descition tree classification
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

# Predicting 
result = regressor.predict([[6.51]])

# Visualizing
plt.title('Position vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.scatter(X, Y)
plt.plot(X, regressor.predict(X))