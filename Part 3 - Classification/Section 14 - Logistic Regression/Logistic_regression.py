import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 14 - Logistic Regression\Social_Network_Ads.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# One Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:4]

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Features Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Using Logistic Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting for a single tuple
result = classifier.predict(sc_X.fit_transform([[1, 30, 87000]]))

# Predicting the test set
Y_pred = classifier.predict(X_test)
comparision = np.concatenate((Y_pred.reshape(-1, 1), Y_test.reshape(-1, 1)), 1)

# Confusion matrix
from sklearn.metrics import confusion_matrix
Correction = confusion_matrix(Y_test, Y_pred)

# Calculating the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred) 