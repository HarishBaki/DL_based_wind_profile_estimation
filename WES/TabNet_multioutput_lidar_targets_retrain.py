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

# === load yaml file and extract variables ===
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
input_file = config['input_file']
input_times_freq = config['input_times_freq']
Coeff_file = config['Coeff_file']
input_variables = config['input_variables']
train_locations = config['train_locations']
train_dates_range = config['train_dates_range']
test_dates_range = config['test_dates_range']
test_locations = config['test_locations']
nEns = config['nEns']
tabnet_param_file = config['tabnet_param_file']
target_variables = config['target_variables']
experiment = config['experiment']
run = 0

# === training and validation data parameters ===#
coeff = xr.open_dataset(Coeff_file)
ERA5 = xr.open_dataset(input_file)

X_train,Y_train,X_valid,Y_valid = data_processing_Heligoland(input_file,Coeff_file,input_times_freq,input_variables,target_variables,train_dates_range,train_locations,val_arg=True)
X_test,Y_test = data_processing_Heligoland(input_file,Coeff_file,input_times_freq,input_variables,target_variables,test_dates_range,train_locations)
print('Training inputs shape:',X_train.shape,'training targets shape:',Y_train.shape)
print('Validation inputs shape:',X_valid.shape,'validation targets shape:',Y_valid.shape)
print('Testing inputs shape:',X_test.shape,'testing targets shape:',Y_test.shape)

OUTPUT_DIR = f'trained_models/{experiment}/run_{run}/Ens_{Ens}'
os.system(f'mkdir -p {OUTPUT_DIR}')

pretrained_experiment = 'ERA5_to_CERRA/17Y'
pretrained_run = 0
pretrained_OUTPUT_DIR = f'trained_models/{pretrained_experiment}/run_{pretrained_run}/Ens_{Ens}'
# Load normalizer
min_max_scaler = joblib.load(f'{pretrained_OUTPUT_DIR}/min_max_scaler.joblib')
# Load model
fSTR = f'TabNet_HOLDOUT_Ens_{str(Ens)}.pkl'
with open(f'{pretrained_OUTPUT_DIR}/{fSTR}', "rb") as f:
    tabReg = pickle.load(f)

# === normalizing the training and validaiton data ---#
Y_train_trans = min_max_scaler.transform(Y_train)
Y_valid_trans = min_max_scaler.transform(Y_valid)
# --- save the normalizing function ---#
joblib.dump(min_max_scaler, f'{OUTPUT_DIR}/min_max_scaler.joblib')

# Retrain model
tabReg.optimizer_params['lr'] = 0.0001
tabReg.fit(X_train=X_train, y_train=Y_train_trans,
                        eval_set=[(X_train, Y_train_trans), (X_valid, Y_valid_trans)],
                        eval_name=['train', 'valid'],
                        max_epochs=250, batch_size=256,    #bSize_opt.item(), 
                        eval_metric=['rmse'], patience=10,  #mae, rmse
                        loss_fn = torch.nn.MSELoss(),
                        warm_start=True,) # the warm_start = True is the key to continue training
    
# Save model
with open(f'{OUTPUT_DIR}/{fSTR}', "wb") as f:
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

Y_pred = tabReg.predict(X_valid)
Y_pred = min_max_scaler.inverse_transform(Y_pred)

for j,target_variable in enumerate(target_variables):
    hexbin_plotter(fig,gs[j+1],Y_valid[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True)
fig.suptitle(f"n_d:{tabReg.n_d}, n_a:{tabReg.n_a}, n_steps:{tabReg.n_steps}, n_independent:{tabReg.n_independent}, n_shared:{tabReg.n_shared}, gamma:{tabReg.gamma}")

plt.savefig(f'{OUTPUT_DIR}/TabNet_HOLDOUT_Ens_{str(Ens)}.png')
plt.close()