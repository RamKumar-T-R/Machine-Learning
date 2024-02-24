import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 17 - Kernel SVM\Social_Network_Ads.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoding', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)[:, 1: 4]

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into train set and test test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Using SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

# Predict for single tuple
result = classifier.predict(sc_X.fit_transform([[1, 30, 87000]]))

# Predicti for whole dataset
Y_pred = classifier.predict(X_test)

# Correction in the model through confusion matrix
from sklearn.metrics import confusion_matrix
correction = confusion_matrix(Y_test, Y_pred)

# Calculating the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)