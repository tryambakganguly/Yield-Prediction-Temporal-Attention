import numpy as np
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

mg_sel = 7

# Results Directory
results_dir = 'results/%s/%s/Tx_%s_run_%s_clusters_%s_hs_%s_dim_%s_dropout_%s_bs_%s_epochs_%s_lr_%s'\
            %(data_type, model, Tx, run_num, n_clusters, h_s, con_dim, dropout, batch_size, epochs, lr_rate)
 
alphas = np.load("%s/y_val_hat_alphas.npy" %(results_dir))   # (10337, 30, 1)
alphas = alphas.reshape((alphas.shape[0], alphas.shape[1]))   # (10337, 30)

results = np.load("%s/y_val.npy" %(results_dir))   # (10337, 7)
# yield, year, location, MG, genotypeID, Cluster, pred_yield

mg_data = results [:, 3]   # (10337, )
mg_index = np.argwhere (mg_data == mg_sel)

yield_actual = results [:, 0]
yield_sel = yield_actual [mg_index]   # (1116, 1)

alphas_sel = alphas [mg_index]   # (1116, 1, 30)
alphas_sel = alphas_sel.reshape ((alphas_sel.shape[0], Tx))   # (1116, 30)

# Attention Weights Yield Wise
def yield_wise_att_weights (lower_limit, upper_limit):
    
    yield_index = np.argwhere((yield_sel > lower_limit) & (yield_sel < upper_limit))   # (12, 2)
    
    alphas_yield = alphas_sel [yield_index[:, 0]]   # (12, 30)
    alphas_yield_mean = np.mean(alphas_yield, axis=0)   # (30, )
    
    return alphas_yield, alphas_yield_mean

alphas_yield_1, alphas_mean_1 = yield_wise_att_weights (np.min(yield_sel), 20)
alphas_yield_2, alphas_mean_2 = yield_wise_att_weights (20, 50)
alphas_yield_3, alphas_mean_3 = yield_wise_att_weights (50, 80)
alphas_yield_4, alphas_mean_4 = yield_wise_att_weights (80, np.max(yield_sel))

# Plot
plt.figure(1)
plt.plot(alphas_mean_1)
plt.plot(alphas_mean_2)
plt.plot(alphas_mean_3)
plt.plot(alphas_mean_4)
plt.ylim (0.020, 0.070)
plt.legend(['Yield < 20 ', '20 < Yield < 50', '50 < Yield < 80', 'Yield > 80'], loc='lower right')
plt.title('Different Ranges of Actual Yield, MG = %s' % (mg_sel))
plt.ylabel('Attention Weights')
plt.xlabel('Timesteps')