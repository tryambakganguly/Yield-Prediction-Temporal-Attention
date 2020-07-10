# Initializations
from preprocess_yield import *    # Check

model = 'lasso'
data_type = 'mg_cluster_weather'
from data_type_func import mg_cluster_weather  # Change
function = mg_cluster_weather # Change
#var_ts = 8   # MG, Cluster, Weather(7)
#var_concat = 1   # MG, Cluster
run_num = 1

# fix random seed for reproducibility
from numpy.random import seed 
seed(run_num)

import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from math import sqrt
import matplotlib.pyplot as plt
import csv
import pandas as pd
from time import time

alpha = 1e-6  # 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1.0

# Create directory to save results
dir_ = 'results/%s/%s/Tx_%s_run_%s_alpha_%s'\
            %(data_type, model, Tx, run_num, alpha)

if not os.path.exists(dir_):
    os.makedirs(dir_)
    
# Data Type
if data_type == 'weather':
    x_train = function(x_train)
    x_val = function(x_val)
    x_test = function(x_test)
else:
    x_train, x_train_mg_cluster = function(x_train, x_train_mg_cluster)
    x_val, x_val_mg_cluster = function(x_val, x_val_mg_cluster)
    x_test, x_test_mg_cluster = function(x_test, x_test_mg_cluster)

# Number of Variables
var_ts = x_train.shape[2]  # MG, Cluster, Weather(7)
var_concat = x_train_mg_cluster.shape[1]   # MG, Cluster

# Print shapes
print('data_train:', x_train.shape, x_train_mg_cluster.shape,\
      'y_train:', y_train.shape, 'yield_train:', yield_train.shape)
print('data_val:', x_val.shape, x_val_mg_cluster.shape,\
      'y_val:', y_val.shape, 'yield_val:', yield_val.shape)
print('data_test:', x_test.shape, x_test_mg_cluster.shape,\
      'y_test:', y_test.shape, 'yield_test:', yield_test.shape)

x_train = x_train.reshape ((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_val = x_val.reshape ((x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
x_test = x_test.reshape ((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Model
model = Lasso(alpha = alpha, random_state = run_num)
start_time = time()    
model.fit(x_train, yield_train)
print("Total train time: ",time()-start_time)

# Plot Ground Truth, Model Prediction
def actual_pred_plot (y_actual, y_pred, n_samples = 60):
    
    # Shape of y_actual, y_pred: (8758, Ty)
    plt.figure()
    plt.plot(y_actual[ : n_samples, -1])  # 60 examples, last prediction time step
    plt.plot(y_pred[ : n_samples, -1])    # 60 examples, last prediction time step
    plt.legend(['Ground Truth', 'Model Prediction'], loc='upper right')
    plt.savefig('%s/actual_pred_plot.png'%(dir_))
    print("Saved actual vs pred plot to disk")
    plt.close()

# Correlation Scatter Plot
def scatter_plot (y_actual, y_pred):
    
    # Shape of y_actual, y_pred: (8758, Ty)
    plt.figure()
    plt.scatter(y_actual[:, -1], y_pred[:, -1])
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=4)
    plt.title('Predicted Value Vs Actual Value')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    #textstr = 'r2_score=  %.3f' %(r2_score(y_actual, y_pred))
    #plt.text(250, 450, textstr, horizontalalignment='center', verticalalignment='top', multialignment='center')
    plt.savefig('%s/scatter_plot.png'%(dir_))
    print("Saved scatter plot to disk")
    plt.close()

# Evaluate Model
def evaluate_model (x_data, yield_data, y_data, states_data, dataset):
    
    # x_train: (82692, 30, 9), x_train_mg_cluster: (82692, 2), yield_train: (82692, 1), y_train: (82692, 6)
    if dataset == "test":
        start_time = time()    
    yield_data_hat = model.predict(x_data)
    if dataset == "test":
        print("Total testing time: ", time()-start_time)  
    
    yield_data_hat = yield_data_hat.reshape((yield_data_hat.shape[0], 1))
    yield_data_hat = scaler_y.inverse_transform(yield_data_hat)
    
    yield_data = scaler_y.inverse_transform(yield_data)
    
    metric_dict = {}  # Dictionary to save the metrics
    
    data_rmse = sqrt(mean_squared_error(yield_data, yield_data_hat))
    metric_dict ['rmse'] = data_rmse 
    print('%s RMSE: %.3f' %(dataset, data_rmse))
    
    data_mae = mean_absolute_error(yield_data, yield_data_hat)
    metric_dict ['mae'] = data_mae
    print('%s MAE: %.3f' %(dataset, data_mae))
    
    data_r2score = r2_score(yield_data, yield_data_hat)
    metric_dict ['r2_score'] = data_r2score
    print('%s r2_score: %.3f' %(dataset, data_r2score))
    
    # Save data
    y_data = np.append(y_data, yield_data_hat, axis = 1)   # (10336, 7)
    np.save("%s/y_%s" %(dir_, dataset), y_data)
    
    # Save States Data
    with open('%s/states_%s.csv' %(dir_, dataset), 'w', newline="") as csv_file:  
        wr = csv.writer(csv_file)
        wr.writerow(states_data)
       
    # Save metrics
    with open('%s/metrics_%s.csv' %(dir_, dataset), 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in metric_dict.items():
            writer.writerow([key, value])
    
    # Save Actual Vs Predicted Plot and Scatter PLot for test set
    if dataset == 'test':
        actual_pred_plot (yield_data, yield_data_hat)
        scatter_plot (yield_data, yield_data_hat)
        
    return metric_dict
   
# Evaluate Model - Train, Validation, Test Sets
train_metrics = evaluate_model (x_train, yield_train, y_train, states_train, 'train')
val_metrics = evaluate_model (x_val, yield_val, y_val, states_val, 'val')
test_metrics = evaluate_model (x_test, yield_test, y_test, states_test, 'test')