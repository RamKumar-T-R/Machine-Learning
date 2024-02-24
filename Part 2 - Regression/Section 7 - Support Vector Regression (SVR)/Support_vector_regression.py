import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the datset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values.reshape(-1, 1)

# Feature Scanling
from sklearn.preprocessing import StandardScaler
sc_X, sc_Y = StandardScaler(), StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Using Support Vector Regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

# Predicting result for 6.5
result = sc_Y.inverse_transform([regressor.predict(sc_X.transform([[6.5]]))])

# Visualing the SVR model
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y))
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X).reshape(-1, 1)))