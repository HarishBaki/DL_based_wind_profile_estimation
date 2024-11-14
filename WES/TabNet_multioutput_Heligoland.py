# === importing dependencies ===#
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import ast
import yaml

import matplotlib.pyplot as plt

from pytorch_tabnet.tab_model import TabNetRegressor
import pickle
from pickle import dump, load
import joblib

import torch

from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

#For reproducibility of the results, the following seeds should be selected 
from numpy.random import seed
randSeed = np.random.randint(1000)

root_dir = '/media/harish/SSD/DL_based_wind_profile_estimation'
sys.path.append(root_dir)
from libraries import *
from plotters import *

# === gather variables provided as input arguments ===
config_file = sys.argv[1] 
Ens = int(sys.argv[2]) #ensemble number
experiment = 'Heligoland_Basu_Coeff'

# === load yaml file and extract variables ===
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
input_file = config['input_file']
input_times_freq = config['input_times_freq']
Coeff_file = config['Coeff_file']
input_variables = config['input_variables']
train_locations = config['train_locations']
test_dates_range = config['test_dates_range']
test_locations = config['test_locations']
nEns = config['nEns']
tabnet_param_file = config['tabnet_param_file']
target_variables = config['target_variables']

# === training and validation data parameters ===#
coeff = xr.open_dataset(Coeff_file)
ERA5 = xr.open_dataset(input_file)

times = coeff.time.values
# === exclude the nan rows from times ===
nan_rows = np.isnan(coeff['data'].values).any(axis=1)
times = times[~nan_rows]

# profle: best fit date, worst fit date
profile_dates = {'High shear': ['2017-10-17T02:00:00','2017-06-06T10:00:00'],
                'Low shear/well mixed':['2017-05-31T11:00:00','2017-11-20T13:00:00'],
                'LLJ': ['2018-04-10T02:00:00','2017-03-30T02:00:00'],
                'High wind': ['2017-12-31T11:00:00','2018-01-05T04:00:00'],
                }
# get the dates into one list
profile_dates_list = []
for key in profile_dates.keys():
    profile_dates_list.append(profile_dates[key][0])
    profile_dates_list.append(profile_dates[key][1])

# convert dates into datetime64[ns]
profile_dates_list = np.array(profile_dates_list, dtype='datetime64[ns]')

# exclude dates from times
times = np.array([time for time in times if time not in profile_dates_list])

# Randomly divide times into training, validation, and testing by 70%, 20%, and 10%
np.random.seed(randSeed)
np.random.shuffle(times)
train_times = times[:int(0.7*len(times))]
valid_times = times[int(0.7*len(times)):int(0.9*len(times))]
test_times = times[int(0.9*len(times)):]
# add the profile_dates_list to test_times
test_times = np.concatenate([test_times, profile_dates_list])

X_train,Y_train = data_processing_Heligoland(input_file,Coeff_file,input_variables,target_variables,train_times,train_locations)
X_valid,Y_valid = data_processing_Heligoland(input_file,Coeff_file,input_variables,target_variables,valid_times,train_locations)
print('Training inputs shape:',X_train.shape,'training targets shape:',Y_train.shape)
print('Validation inputs shape:',X_valid.shape,'validation targets shape:',Y_valid.shape)

# === normalizing the training and validaiton data ---#
min_max_scaler = preprocessing.MinMaxScaler().fit(Y_train)

Y_train_trans = min_max_scaler.transform(Y_train)
Y_valid_trans = min_max_scaler.transform(Y_valid)

OUTPUT_DIR = f'trained_models/{experiment}/Ens_{Ens}'
os.system(f'mkdir -p {OUTPUT_DIR}')
# --- save the normalizing function ---#
joblib.dump(min_max_scaler, f'{OUTPUT_DIR}/min_max_scaler.joblib')

# === load tabnet parameters ===
tabnet_params = pd.read_csv(tabnet_param_file)
n_d = int(tabnet_params['n_d'][Ens])
n_a = int(tabnet_params['n_a'][Ens])
n_steps = int(tabnet_params['n_steps'][Ens])
n_independent = int(tabnet_params['n_independent'][Ens])
n_shared = int(tabnet_params['n_shared'][Ens])
gamma = float(tabnet_params['gamma'][Ens])

# === training the tabnet model ===#
tabReg   = TabNetRegressor(n_d = n_d, 
                                n_a = n_a, 
                                n_steps = n_steps,
                                n_independent = n_independent,
                                n_shared = n_shared,
                                gamma = gamma,
                                verbose=1,seed=randSeed, )

tabReg.fit(X_train=X_train, y_train=Y_train_trans,
                    eval_set=[(X_train, Y_train_trans), (X_valid, Y_valid_trans)],
                    eval_name=['train', 'valid'],
                    max_epochs=250, batch_size=512,    #bSize_opt.item(), 
                    eval_metric=['rmse'], patience=10,  #mae, rmse
                    loss_fn = torch.nn.MSELoss())

fSTR = f'{OUTPUT_DIR}/TabNet_HOLDOUT_Ens_{str(Ens)}.pkl'
with open(fSTR, "wb") as f:
    dump(tabReg, f, pickle.HIGHEST_PROTOCOL)
print('dumped')

# --- Plot loss curve and hexbin ---
fig = plt.figure(figsize=(18, 3), constrained_layout=True)
gs = fig.add_gridspec(1,6)

# Line plot for train and validation RMSE
ax = fig.add_subplot(gs[0])
ax.plot(tabReg.history['train_rmse'],'--', label='train')
ax.plot(tabReg.history['valid_rmse'],':', label='validation')
ax.set_title('Training and Validation RMSE')
ax.set_xlabel('Epochs')
ax.set_ylabel('RMSE')
ax.legend()

Y_pred = min_max_scaler.inverse_transform(tabReg.predict(X_valid))
for j,target_variable in enumerate(target_variables):
    hexbin_plotter(fig,gs[j+1],Y_valid[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True)
fig.suptitle(f"n_d:{n_d}, n_a:{n_a}, n_steps:{n_steps}, n_independent:{n_independent}, n_shared:{n_shared}, gamma:{gamma}")

plt.savefig(f'{OUTPUT_DIR}/TabNet_HOLDOUT_Ens_{str(Ens)}.png')
plt.close()