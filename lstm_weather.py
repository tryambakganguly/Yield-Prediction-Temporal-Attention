# Initializations
from preprocess_yield import *    # Check

model = 'lstm'
data_type = 'weather'
from data_type_func import weather  # Change
function = weather # Change
#var_ts = 8   # MG, Cluster, Weather(7)
#var_concat = 1   # MG, Cluster
run_num = 1

# fix random seed for reproducibility
from numpy.random import seed 
seed(run_num)
from tensorflow import set_random_seed
set_random_seed(run_num)

import os
import numpy as np
from keras.layers import Concatenate, Dot, Input, LSTM, RepeatVector, Dense
from keras.layers import Dropout, Flatten, Reshape, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.activations import softmax
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import csv
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #gpu_number=2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

h_s = 128   # {32, 64, 96, 128, 256}
dropout = 0.2  
batch_size = 512  
epochs = 100  # 100
lr_rate = 0.001   # (0.001, 3e-4, 5e-4)
con_dim = 2   # (1, 2, 4, 8, 16) # Reduction in dimension of the temporal context to con_dim before concat with MG, Cluster

# Create directory to save results
dir_ = 'results/%s/%s/Tx_%s_run_%s_clusters_%s_hs_%s_dim_%s_dropout_%s_bs_%s_epochs_%s_lr_%s'\
            %(data_type, model, Tx, run_num, n_clusters, h_s, con_dim, dropout, batch_size, epochs, lr_rate)

if not os.path.exists(dir_):
    os.makedirs(dir_)
    
# Data Type
x_train = function(x_train)
x_val = function(x_val)
x_test = function(x_test)

# Number of Variables
var_ts = x_train.shape[2]  # MG, Cluster, Weather(7)
#var_concat = x_train_mg_cluster.shape[1]   # MG, Cluster

# Print shapes
print('data_train:', x_train.shape, 'y_train:', y_train.shape, 'yield_train:', yield_train.shape)
print('data_val:', x_val.shape, 'y_val:', y_val.shape, 'yield_val:', yield_val.shape)
print('data_test:', x_test.shape, 'y_test:', y_test.shape, 'yield_test:', yield_test.shape)

# Model
def model(Tx, var_ts, h_s, dropout):
    
    # Tx : Number of input timesteps
    # var_ts: Number of input variables
    # h_s: Hidden State Dimensions for Encoder, Decoder
    encoder_input = Input(shape = (Tx, var_ts))   # (None, 10, 8)
    
    # Encoder LSTM      
    lstm_1, state_h, state_c = LSTM(h_s, return_state=True, return_sequences=True)(encoder_input)
    lstm_1 = Dropout (dropout)(lstm_1)     # (None, 10, 16)
    
    lstm_2, state_h, state_c = LSTM(h_s, return_state=True, return_sequences=False)(lstm_1)
    lstm_2 = Dropout (dropout)(lstm_2)     # (None, 16)
    
    # FC Layer
    yhat = Dense (1, activation = "linear")(lstm_2)
        
    pred_model = Model(encoder_input, yhat)   # Prediction Model
        
    return pred_model

            
# Model Summary
pred_model = model(Tx, var_ts, h_s, dropout)
pred_model.summary()

# Train Model
pred_model.compile(loss='mean_squared_error', optimizer = Adam(lr=lr_rate)) 

hist = pred_model.fit (x_train, yield_train,
                  batch_size = batch_size,
                  epochs = epochs,
                  #callbacks = callback_lists,   # Try Early Stopping
                  verbose = 2,
                  shuffle = True,
                  validation_data=(x_val, yield_val))

# Plot
loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.savefig('%s/loss_plot.png'%(dir_))
print("Saved loss plot to disk") 
plt.close()

# Save Data
loss = pd.DataFrame(loss).to_csv('%s/loss.csv'%(dir_))    # Not in original scale 
val_loss = pd.DataFrame(val_loss).to_csv('%s/val_loss.csv'%(dir_))  # Not in original scale

# Plot Ground Truth, Model Prediction
def actual_pred_plot (y_actual, y_pred, n_samples = 60):
    
    # Shape of y_actual, y_pred: (10337, 1)
    plt.figure()
    plt.plot(y_actual[ : n_samples])  # 60 examples
    plt.plot(y_pred[ : n_samples])    # 60 examples
    plt.legend(['Ground Truth', 'Model Prediction'], loc='upper right')
    plt.savefig('%s/actual_pred_plot.png'%(dir_))
    print("Saved actual vs pred plot to disk")
    plt.close()

# Correlation Scatter Plot
def scatter_plot (y_actual, y_pred):
    
    # Shape of y_actual, y_pred: (10337, 1)
    plt.figure()
    plt.scatter(y_actual[:], y_pred[:])
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
    yield_data_hat = pred_model.predict(x_data, batch_size = batch_size)
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