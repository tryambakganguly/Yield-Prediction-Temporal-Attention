# fix random seed for reproducibility
from numpy.random import seed 
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np

# data_x: (n_samples, 30, 9), MG: 0, Cluster: 1, Weather: [2:]
# data_x_mg_cluster: (n_samples, 2), MG: 0, Cluster: 1

# MG, Cluster, Weather
def mg_cluster_weather(data_x, data_x_mg_cluster):
    
    data_x = data_x
    data_x_mg_cluster = data_x_mg_cluster
    
    return data_x, data_x_mg_cluster
    
# MG, Cluster
def mg_cluster(data_x, data_x_mg_cluster):
    
    data_x = data_x [:, :, 0 : 2]   # (n_samples, 30, 2)
    data_x_mg_cluster = data_x_mg_cluster
    
    return data_x, data_x_mg_cluster

# MG
def mg(data_x, data_x_mg_cluster):
    
    data_x = data_x [:, :, 0]
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))   # (n_samples, 30, 1)
    
    data_x_mg = data_x_mg_cluster [:, 0]   # (n_samples, )
    data_x_mg = data_x_mg.reshape((data_x_mg.shape[0], 1))   # (n_samples, 1)
    
    return data_x, data_x_mg

# Cluster
def cluster(data_x, data_x_mg_cluster):
    
    data_x = data_x [:, :, 1]
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))   # (n_samples, 30, 1)
    
    data_x_cluster = data_x_mg_cluster [:, 1]   # (n_samples, )
    data_x_cluster = data_x_cluster.reshape((data_x_cluster.shape[0], 1))   # (n_samples, 1)
    
    return data_x, data_x_cluster

# data_x: (n_samples, 30, 9), MG: 0, Cluster: 1, Weather: [2:]
# data_x_mg_cluster: (n_samples, 2), MG: 0, Cluster: 1
    
# MG, Weather
def mg_weather(data_x, data_x_mg_cluster):
    
    data_x = np.delete(data_x, 1, axis = 2)   # Remove Cluster Column, (n_samples, 30, 8)
    
    data_x_mg = data_x_mg_cluster [:, 0]   # (n_samples, )
    data_x_mg = data_x_mg.reshape((data_x_mg.shape[0], 1))   # (n_samples, 1)
    
    return data_x, data_x_mg

# Cluster, Weather
def cluster_weather(data_x, data_x_mg_cluster):
    
    data_x = np.delete(data_x, 0, axis = 2)   # Remove MG Column, (n_samples, 30, 8)
    
    data_x_cluster = data_x_mg_cluster [:, 1]   # (n_samples, )
    data_x_cluster = data_x_cluster.reshape((data_x_cluster.shape[0], 1))   # (n_samples, 1)
    
    return data_x, data_x_cluster

# Weather
def weather(data_x):
    
    data_x = data_x [:, :, 2 : ]   # (n_samples, 30, 7)
    
    return data_x    

