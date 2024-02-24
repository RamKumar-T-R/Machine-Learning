import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 24 - K-Means Clustering\Mall_Customers.csv')
X = dataset.iloc[:, 3:].values

# Using elbow method by considering k-means class
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)

# Plotting the WCSS values to find out the knee point
plt.title('Knee point')
plt.ylabel('WCSS')
plt.xlabel('No. of Clusters')
plt.plot(range(1, 11), wcss)

# Using KMeans model with optimal no. of clusters from elbow method
cluster = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
cluster.fit_transform(X)
Y_kmeans = cluster.predict(X)

"""
# Visualizing the model
# cluster 1
plt.scatter(X[cluster == 0 , 0], X[cluster == 0, 1], s = 100, c = 'red', label = 'Cluster1')
plt.scatter(X[cluster == 1 , 0], X[cluster == 1, 1], s = 100, c = 'blue', label = 'Cluster2')
plt.scatter(X[cluster == 2 , 0], X[cluster == 2, 1], s = 100, c = 'green', label = 'Cluster3')
plt.scatter(X[cluster == 3 , 0], X[cluster == 3, 1], s = 100, c = 'yellow', label = 'Cluster4')
plt.scatter(X[cluster == 4 , 0], X[cluster == 4, 1], s = 100, c = 'grey', label = 'Cluster5')
plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Clusters of customers')
"""