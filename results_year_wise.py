import numpy as np

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

# Results Directory
results_dir = 'results/%s/%s/Tx_%s_run_%s_clusters_%s_hs_%s_dim_%s_dropout_%s_bs_%s_epochs_%s_lr_%s'\
            %(data_type, model, Tx, run_num, n_clusters, h_s, con_dim, dropout, batch_size, epochs, lr_rate)

results = np.load("%s/y_test.npy" %(results_dir))   # (10337, 7)
# yield, year, location, MG, genotypeID, Cluster, pred_yield

year_data = results [:, 1]
yield_actual = results [:, 0]
yield_pred = results [:, 6]

mae_dict = {}  # Dictionary to save MAE for different years
    
for year in range(2003, 2016):
    
    year_index = np.argwhere (year_data == year)
    actual_mean = np.mean(yield_actual [year_index])   # yield actual mean for the year
    pred_mean = np.mean(yield_pred [year_index])       # yield pred mean for the year
    
    data_mae = np.absolute(actual_mean - pred_mean)
    mae_dict [year] = data_mae
    
    print(year,'Actual Yield:',round(actual_mean,3),'Pred Yield:',round(pred_mean,3),\
                      'Error:',round(data_mae, 3))        
    