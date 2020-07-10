from sklearn.cluster import KMeans
from pandas import read_csv
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

n_clusters_range = [2, 3, 4, 5, 10, 15, 20, 25]
normalization = 'False'    # 'True', 'False'
random_state = 1   # {1, 2, 3}
algo = 'full'   # 'full', 'elkan'

data_file = 'data/USTrelmatrix_GenoID.csv'

# Correlation Matrix
corr_matrix = read_csv(data_file, header=0, index_col=0)   # (5839, 5839)
corr_matrix = corr_matrix.values   # (5839, 5839)

# Normalize
if normalization == 'True':
    corr_matrix = normalize(corr_matrix)

# Choosing Optimal Number of Clusters
sum_squared_dist = []   # Empty List
cluster_centers = []   # Empty List

# Fit k-means for different number of clusters
for n_clusters in n_clusters_range:
    
    k_means = KMeans(n_clusters = n_clusters, algorithm = algo, random_state = random_state)
    t0 = time.time()
    kmeans = k_means.fit(corr_matrix)
    sum_squared_dist.append(kmeans.inertia_)
    cluster_centers.append(kmeans.cluster_centers_)
    
    print("Number of clusters", n_clusters, "Time Taken", time.time()-t0)

# Plot Loss Vs Clusters
plt.plot(n_clusters_range, sum_squared_dist, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Sum of Squared Distances')
plt.title('Loss Vs Clusters Used')