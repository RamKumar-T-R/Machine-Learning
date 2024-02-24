import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 6 - Polynomial Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1].values
Y = dataset.iloc[:, 2].values

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
simpleRegressor = LinearRegression()
simpleRegressor.fit(X.reshape(-1, 1), Y.reshape(-1, 1))

# Polynomial Linear Regression

# Creating a poly
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4) # degree is the highest power of the polynomial
X_poly = poly.fit_transform(X.reshape(-1, 1))

# Simple Linear Regression with X_ploy (n features) and Y
polynomialRegressor = LinearRegression()
polynomialRegressor.fit(X_poly, Y)

# Visualizing Linear Regression
plt.title('Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.scatter(X, Y)
plt.plot(X, simpleRegressor.predict(X.reshape(-1, 1)), color = 'red')

# Visualizing Plynomial Regression
plt.title('Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.scatter(X, Y)
plt.plot(X, polynomialRegressor.predict(X_poly), color = 'red')

# Predicting with linear Regression
linear_pred = simpleRegressor.predict([[6]])

# Predicting with Polynomia Regression
polynomia_pred = polynomialRegressor.predict(poly.fit_transform([[6]]))