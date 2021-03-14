# Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same dataset for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

X = pd.read_csv('em-kmeans.csv')
x1 = X['Distance_Feature'].values
x2 = X['Speeding_Feature'].values
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

plt.plot()
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

gm = GaussianMixture(n_components = 3)
gm.fit(X)
em = gm.predict(X)

print("EM values:")
print(em)
print()
print("Means:")
print(gm.means_)
print()
print("Covariances:")
print(gm.covariances_)
print()
print("Input:")
print(X)
print()

plt.title('Gaussian Mixture')
plt.scatter(X[:,0], X[:,1], c = em, s = 50)
plt.show()

km = KMeans(n_clusters = 3)
km.fit(X)

print("Cluster centers:")
print(km.cluster_centers_)
print()
print("Labels:")
print(km.labels_)

plt.title('KMeans')
plt.scatter(X[:,0], X[:,1], c = km.labels_, cmap = 'rainbow')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'black')
plt.show()
