import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 25 - Hierarchical Clustering\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

from scipy.cluster import hierarchy
dendogram = hierarchy.dendrogram(hierarchy.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Points')
plt.ylabel('Euclidian Distance')

# Using Hierarchical Agglomeration Clustering 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Y = cluster.fit_predict(X)

# Visulizing the clusters
"""
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