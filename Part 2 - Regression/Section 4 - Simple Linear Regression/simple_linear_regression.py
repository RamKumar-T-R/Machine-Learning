import pandas as pd
import matplotlib.pyplot as plt
# Importing dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv')
print(dataset)
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Using Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train.reshape(-1, 1), y_train) # it expects only the 2D array

# Predicting
y_pred = regressor.predict(x_test.reshape(-1, 1))

# Visualizing
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.plot(x_train, regressor.predict(x_train.reshape(-1, 1)), color = 'blue')
plt.scatter(x_train, y_train, color = 'red')

plt.title("Salary vs Expericence (Test set)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train.reshape(-1, 1)), color = 'blue')
plt.scatter([[12]], )  
