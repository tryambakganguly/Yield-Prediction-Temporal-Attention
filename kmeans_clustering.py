from sklearn.cluster import KMeans
from pandas import read_csv
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn import linear_model
import numpy as np

n_clusters = 20
random_state = 1   # {1, 2, 3}
normalization = 'False'    # 'True', 'False'
algo = 'full'   # 'full', 'elkan'

dir_ = 'data'
data_file = 'data/USTrelmatrix_GenoID.csv'

# Correlation Matrix
corr_matrix = read_csv(data_file, header=0, index_col=0)   # (5839, 5839)
corr_matrix = corr_matrix.values   # (5839, 5839)

# Normalize
if normalization == 'True':
    corr_matrix = normalize(corr_matrix)

# K-Means Clustering    
k_means = KMeans(n_clusters = n_clusters, algorithm = algo, random_state = random_state)
t0 = time.time()
kmeans = k_means.fit(corr_matrix)
print("Number of clusters", n_clusters, "Time Taken", time.time()-t0)

clusters = kmeans.predict(corr_matrix)   # (5839, )           

# Cluster Cardinality, Number of examples per cluster
unique_clusters, counts_clusters = np.unique(clusters, return_counts=True)

plt.figure(1)
plt.bar(unique_clusters, counts_clusters, width=0.8, align='center')
plt.title('Cluster Cardinality')
plt.ylabel('Points in Cluster')
plt.xlabel('Cluster Number')

# Cluster Magnitude, Sum of distances from all examples to the centroid of the cluster
all_dist = kmeans.transform(corr_matrix)   # (5839, 20)   Distance to all cluster centres for each data point
cluster_dist = np.min(all_dist, axis = 1)  # (5839, )  Min distance is the distance to the closest centre

cluster_magnitude = np.zeros ((n_clusters, ))   # (20, )
     
for i in range(n_clusters):
    index = np.argwhere(clusters == i)     # Index corresponding to the cluster ID
    dist_sum = np.sum((cluster_dist[index]))   # Extracting all distances and taking sum
    cluster_magnitude [i] = dist_sum

plt.figure(2)
plt.bar(unique_clusters, cluster_magnitude, width=0.8, align='center')
plt.title('Cluster Magnitude')
plt.ylabel('Total Point-to-Centroid distance')
plt.xlabel('Cluster Number')


# Magnitude vs. Cardinality
regr = linear_model.LinearRegression()     # Create linear regression object
counts_clusters = counts_clusters.reshape((counts_clusters.shape[0], 1))   # (20, 1)
regr.fit(counts_clusters, cluster_magnitude)
y_pred = regr.predict(counts_clusters)

plt.figure(3)
plt.scatter(counts_clusters, cluster_magnitude)
plt.plot(counts_clusters, y_pred, color='blue', linewidth=3)
plt.title('Magnitude vs Cardinality')
plt.ylabel('Magnitude')
plt.xlabel('Cardinality')

# Save
np.save("%s/k_means_clusters_%s"%(dir_, n_clusters), clusters)
