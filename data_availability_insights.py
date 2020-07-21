import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

model = 'lstm_att'
data_type = 'mg_cluster_weather'
run_num = 1

Tx = 30
n_clusters = 20

h_s = 128   
dropout = 0.2  
batch_size = 512  
epochs = 100   
lr_rate = 0.001   
con_dim = 2   

threshold = 50    # combinations having this minimum number of test samples to be considered

# Results Directory
results_dir = 'results/%s/%s/Tx_%s_run_%s_clusters_%s_hs_%s_dim_%s_dropout_%s_bs_%s_epochs_%s_lr_%s'\
            %(data_type, model, Tx, run_num, n_clusters, h_s, con_dim, dropout, batch_size, epochs, lr_rate)

results_train = np.load("%s/y_train.npy" %(results_dir))   # (82692, 7)
# yield, year, location, MG, genotypeID, Cluster, pred_yield

results_test = np.load("%s/y_test.npy" %(results_dir))    # (10337, 7)
# yield, year, location, MG, genotypeID, Cluster, pred_yield

train_location = results_train [:, 2]
train_mg = results_train [:, 3]
train_cluster = results_train [:, 5]

test_location = results_test [:, 2]
test_mg = results_test [:, 3]
test_cluster = results_test [:, 5]

# Number of Unique Values
test_location_list = np.unique(test_location, return_counts = False)    # 158 values
test_mg_list = np.unique(test_mg, return_counts = False)    # 10 values
test_cluster_list = np.unique(test_cluster, return_counts = False)    # 20 values

# Cluster Vs MG, Test RMSE
cluster_mg_rmse = np.zeros((test_mg_list.shape[0], test_cluster_list.shape[0]))    # (10, 20)

# Number of samples in train set/Number of unique locations
cluster_mg_ratio = np.zeros((test_mg_list.shape[0], test_cluster_list.shape[0]))    # (10, 20)

for row in range (0, test_mg_list.shape[0]):
    mg = test_mg_list [row]
    
    for column in range (0, test_cluster_list.shape[0]):
        cluster = test_cluster_list [column]
        
        # Extract test set for the cluster, mg combination
        results_test_sel = results_test [(results_test[:, 5] == cluster) & (results_test[:, 3] == mg)]
        
        # When number of test samples very less, ignore the combinations
        if results_test_sel.shape[0] <= threshold:
            cluster_mg_rmse [row, column] = np.nan
            cluster_mg_ratio [row, column] = np.nan
            
        # Consider combinations having samples more than threshold
        if results_test_sel.shape[0] > threshold:
            
            # Number of Samples in Train Set/Number of Unique Locations
            results_train_sel = results_train [(results_train[:, 5] == cluster) & (results_train[:, 3] == mg)]
            num_samples_train = results_train_sel.shape[0]
            
            train_location_sel = results_train_sel [:, 2]
            unique_locations = np.unique(train_location_sel, return_counts=False)
            
            cluster_mg_ratio [row, column] = num_samples_train/unique_locations.shape[0]
            
            # Test RMSE for the combination
            yield_actual = results_test_sel [:, 0]
            yield_pred = results_test_sel [:, 6]
            cluster_mg_rmse [row, column] = sqrt(mean_squared_error(yield_actual, yield_pred))
            
            

# Number of Samples in Training Set/Number of Unique Locations, Test RMSE - for Cluster, MG pair    
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(cluster_mg_ratio)
plt.title('Number of Samples in Training Set/Number of Unique Locations')
plt.xlabel('Genotype Cluster')
plt.ylabel('Maturity Group')
plt.colorbar(orientation='horizontal')
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(cluster_mg_rmse)
plt.title('Test RMSE')
plt.xlabel('Genotype Cluster')
plt.ylabel('Maturity Group')
plt.colorbar(orientation='horizontal')