import numpy as np
import pandas as pd

data_dir="Data_Reframed"

print("Data Loading")
X=np.load("%s/mg_weather_variables_other_test_data_TS_214_days.npy"%(data_dir))

X= X[:, 0:210, :]  # 4 days dropped 

n_samples = X.shape[0]     # Total no of samples
n_timesteps = X.shape[1]   # No of time steps
n_variables = X.shape[2]   # No of input variables

# Variables (Column Wise)
# MG 
# Average Direct Normal Irradiance
# Average Precipitation Previous Hour (inches)
# Average Relative Humidity
# Maximum Direct Normal Irradiance
# Maximum Surface Temperature 
# Minimum Surface Temperature
# Average Surface Temperature (Fahrenheit)

# Weekly 
data_interval_1= 7
n_timesteps_1 = n_timesteps/ data_interval_1    # 30

X_weekly = np.zeros((n_samples, 30, n_variables)) 

# Bi-weekly
data_interval_2= 14
n_timesteps_2 = n_timesteps/ data_interval_2   # 15

X_biweekly = np.zeros((n_samples, 15, n_variables)) 

# Monthly 
data_interval_3= 30
n_timesteps_3 = n_timesteps/ data_interval_3  # 7

X_monthly = np.zeros((n_samples, 7, n_variables)) 


def data_reshape(dataset, time_interval):   # dataset (210, 8) 
    
    nb_timesteps = 210/time_interval    # 210 / 7 = 30
    nb_timesteps = int(nb_timesteps)
    
    data_reshaped = np.zeros ((nb_timesteps, 8))   # (30, 8) # n_variables = 8
    
    
    for i in range(0, nb_timesteps):
        range_1 = i * time_interval
        range_2 = (i + 1) * time_interval
        
        data = dataset [range_1:range_2, :]  # (7, 8)
        
        data_reshaped [i, 0] = np.mean (data[:, 0], axis=0)   # MG 
        data_reshaped [i, 1] = np.mean (data[:, 1], axis=0)   # Average Direct Normal Irradiance
        data_reshaped [i, 2] = np.mean (data[:, 2], axis=0)   # Average Precipitation Previous Hour (inches)
        data_reshaped [i, 3] = np.mean (data[:, 3], axis=0)   # Average Relative Humidity
        data_reshaped [i, 4] =  np.amax (data[:, 4], axis=0)  # Maximum Direct Normal Irradiance
        data_reshaped [i, 5] = np.amax (data[:, 5], axis=0)   # Maximum Surface Temperature 
        data_reshaped [i, 6] = np.amin (data[:, 6], axis=0)   # Minimum Surface Temperature
        data_reshaped [i, 7] = np.mean (data[:, 7], axis=0)   # Average Surface Temperature (Fahrenheit)
        
              
                       
    return data_reshaped


for count in range(0, n_samples):
    
    X_count = X [count, :, :]
    
    X_weekly [count, :, :] = data_reshape (X_count, 7)
    X_biweekly [count, :, :] = data_reshape (X_count, 14)
    X_monthly [count, :, :] = data_reshape (X_count, 30)
    print(count)        
  
np.save("%s/mg_weather_variables_other_test_data_TS_30_weeks_prep_avg"%(data_dir), X_weekly)
np.save("%s/mg_weather_variables_other_test_data_TS_15_biweekly_prep_avg"%(data_dir), X_biweekly)
np.save("%s/mg_weather_variables_other_test_data_TS_7_months_prep_avg"%(data_dir), X_monthly)
