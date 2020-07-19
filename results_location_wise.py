import numpy as np
from sklearn.metrics import mean_absolute_error

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

location = 2   # Ames
year = 2003

# Results Directory
results_dir = 'results/%s/%s/Tx_%s_run_%s_clusters_%s_hs_%s_dim_%s_dropout_%s_bs_%s_epochs_%s_lr_%s'\
            %(data_type, model, Tx, run_num, n_clusters, h_s, con_dim, dropout, batch_size, epochs, lr_rate)

results = np.load("%s/y_test.npy" %(results_dir))   # (10337, 7)
# yield, year, location, MG, genotypeID, Cluster, pred_yield

loc_data = results [:, 2]
year_data = results [:, 1]
loc_index = np.argwhere ((results [:, 2] == location) & (results [:, 1] == year))

yield_actual = results [:, 0]
yield_pred = results [:, 6]

yield_actual_loc = yield_actual [loc_index]
yield_pred_loc = yield_pred [loc_index]

print ("Maximum Yield:", np.max(yield_actual_loc))
print ("Minimum Yield:", np.min(yield_actual_loc))
print ("MAE:", mean_absolute_error(yield_actual_loc, yield_pred_loc))      
    
    