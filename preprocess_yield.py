# fix random seed for reproducibility
from numpy.random import seed 
seed(1)

import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

Tx = 30
n_var = 9   # "all" : 9, "weather" : 7
n_clusters = 20
val_size = 0.1
test_size = 0.1
data_dir = "data_subset"

# Weather Variables, MG, Yield, Year, Location
x_data = np.load("%s/mg_weather_variables_all_data_TS_%s.npy" %(data_dir, Tx))   # (103365, 30, 8)
y_data = np.load("%s/yield_year_location_all_data.npy" %(data_dir))   # (103365, 3) yield, year, location

# K-Means Clusters (Genotype)
kmeans_clusters = np.load("%s/k_means_clusters_%s.npy" %(data_dir, n_clusters))   # (5839, )

# State
state_data = read_csv("%s/state_list_all_data.csv" %(data_dir), header=0, index_col=0)   # (103365, 1)
state_data = state_data.values.tolist()

# Genotype ID
genotypeID = np.load("%s/genotype_id_all_data.npy"%(data_dir))   # (103365, 1) max(genotype ID) = 5839

# Genotype Cluster Sample Wise
x_data_cluster = np.zeros((genotypeID.shape[0], 1))   # (103365, 1)
for i in range(0, genotypeID.shape[0]):
    id_genotype = genotypeID[i]
    x_data_cluster[i] = kmeans_clusters [int(id_genotype - 1)]   # clusters [5838]
    
# Different Variables (Total- 8)
# Column (0)- MG
# Column (1)- Average Direct Normal Irradiance (ADNI)
# Column (2)- Average Precipitation Previous Hour (inches) (AP)
# Column (3)- Average Relative Humidity (ARH)
# Column (4)- Maximum Direct Normal Irradiance (MDNI)
# Column (5)- Maximum Surface Temperature (MaxSur)
# Column (6)- Minimum Surface Temperature (MinSur)
# Column (7)- Average Surface Temperature (Fahrenheit) (AvgSur)

# MG
x_data_mg = x_data [:, 0, 0]   # (103365, )
x_data_mg = x_data_mg.reshape((x_data_mg.shape[0], 1))   # (103365, 1)

# Append
y_data = np.append(y_data, x_data_mg, axis=1)   # (103365, 4) yield, year, location, MG
y_data = np.append(y_data, genotypeID, axis=1)   # (103365, 5) yield, year, location, MG, genotypeID
y_data = np.append(y_data, x_data_cluster, axis=1)   # (103365, 6) yield, year, location, MG, genotypeID, Cluster

# All Input Variables
x_variables = np.zeros((x_data.shape[0], x_data.shape[1], n_var))   # (103365, 30, 9)

# MG
x_variables [:, :, 0] = x_data [:, :, 0]   

# Cluster
for i in range(0, x_variables.shape[0]):
    x_variables[i, :, 1] = x_data_cluster[i]

# Weather Variables
x_variables[:, :, 2 :] = x_data [:, :, 1 :]   

# Range of Indices
indices = range (x_variables.shape[0])

# Train, Validation and Test Sets
x_train, x_val_test, y_train, y_val_test, in_train, in_val_test\
        = train_test_split(x_variables, y_data, indices, test_size = (val_size + test_size), random_state=42)

# Validation, Test Sets
x_val, x_test, y_val, y_test, in_val, in_test = train_test_split(x_val_test, y_val_test, in_val_test,\
                                                test_size=(test_size/(val_size+test_size)), random_state=42)       

# State (Train, Validation, Test Sets)
states_train = [state_data[index] for index in in_train]   # 82692 elements
states_val = [state_data[index] for index in in_val]   # 10336 elements
states_test = [state_data[index] for index in in_test]   # 10337 elements

# Scale features 
# Two separate scalers for X, Y (diff dimensions) 
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y =  MinMaxScaler(feature_range=(-1, 1))

x_train_reshaped = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

yield_train = y_train [:, 0]   # (82692, )
yield_train = yield_train.reshape((yield_train.shape[0], 1))   # (82692, 1)

# Scaling Coefficients calculated from the training dataset
scaler_x = scaler_x.fit(x_train_reshaped)   
scaler_y = scaler_y.fit(yield_train)

# Function to scale features after fitting
def scale_features (data_x, data_y):
    
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2]))
    data_x = scaler_x.transform(data_x)
    data_x = data_x.reshape((data_x.shape[0], Tx, n_var))
    
    data_x_mg = data_x [:, 0, 0]
    data_x_mg = data_x_mg.reshape((data_x_mg.shape[0], 1))   # (82692, 1)
    data_x_cluster = data_x [:, 0, 1]
    data_x_cluster = data_x_cluster.reshape((data_x_cluster.shape[0], 1))   # (82692, 1)
    data_x_mg_cluster = np.append (data_x_mg, data_x_cluster, axis=1)   # (82692, 2)
    
    data_yield = data_y [:, 0]
    data_yield = data_yield.reshape((data_yield.shape[0], 1))
    data_yield = scaler_y.transform(data_yield)
    
    return data_x, data_x_mg_cluster, data_yield

# Scale features
x_train, x_train_mg_cluster, yield_train = scale_features (x_train, y_train)   # (82692,30,9), (82692,2), (82692, 1)
x_val, x_val_mg_cluster, yield_val = scale_features (x_val, y_val)   # (10336, 30, 9), (10336, 2), (10336, 1)
x_test, x_test_mg_cluster, yield_test = scale_features (x_test, y_test)   # (10337, 30, 9), (10337, 2), (10337, 1)