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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as R2

from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

#For reproducibility of the results, the following seeds should be selected 
from numpy.random import seed
#seed(20) #comment this, so that different samplings are created each ensemble
randSeed = np.random.randint(1000)

# === gather variables provided as input arguments ===
config_file = sys.argv[1]
target_variables = ast.literal_eval(sys.argv[2])
Ens = int(sys.argv[3])
print(config_file,target_variables,Ens)

# === load yaml file and extract variables ===
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
ERA_file = config['ERA_file']
ChSh_Coeff_file = config['ChSh_Coeff_file']
input_variables = config['input_variables']
train_dates_range = config['train_dates_range']
train_locations = config['train_locations']
n_d = config['n_d']
n_steps = config['n_steps']
n_independent = config['n_independent']
n_shared = config['n_shared']
gamma = config['gamma']
nTrial = config['nTrial']
nEns = config['nEns']
feature_importance = config['feature_importance']
number_of_features = config['number_of_features']
experiment = config['experiment']

# === Load important features ===#
if feature_importance[0]:
    sorted_feature_importance_array = np.load(f'Coefficient_{target_variables[0]}_featImp.npy')    
    # Access the data from the numpy array
    feature_names = sorted_feature_importance_array['Feature']
    importances = sorted_feature_importance_array['Importance']
    print(feature_names)
    input_variables = feature_names[:number_of_features[target_variables[0]]]


# === Datasets ===#
ERA5 = xr.open_dataset(ERA_file)
ChSh_Coeff = xr.open_dataset(ChSh_Coeff_file)

def data_processing(input_variables,target_variables, dates_range, locations,test_arg=None):
    ERA5 = xr.open_dataset('data/ERA5.nc')
    ChSh_Coeff = xr.open_dataset('data/Chebyshev_Coefficnents.nc')

    #=== Extracting training and validation indices ===# 
    time_coord = ERA5.sel(time=slice(*dates_range)).coords['time']
    years = time_coord.dt.year
    months = time_coord.dt.month
    validation_times = np.zeros(len(time_coord), dtype=bool)
    for year in np.unique(years):
        for month in range(1, 13):
            month_indices = np.where((years == year) & (months == month))[0]
            if len(month_indices) >= 144:
                start_index = np.random.choice(len(month_indices) - 143)
                validation_indices = month_indices[start_index:start_index + 144]
                validation_times[validation_indices] = True
        
    if test_arg:
        X_test = np.empty((0, len(input_variables)))
        Y_test = np.empty((0, len(target_variables)))

        for loc in locations:
            # --- testing ---#
            X_loc = ERA5[input_variables].sel(time=slice(*dates_range)).sel(location=loc).to_array().values.T
            X_test = np.concatenate((X_test, X_loc), axis=0)
            Y_loc = ChSh_Coeff.sel(time=slice(*dates_range)).sel(coeff=target_variables, obs=loc).to_array().values
            Y_test = np.concatenate((Y_test, Y_loc[0,:,:]), axis=0)

        # Replace NaN values with zeros
        X_test = np.nan_to_num(X_test)
        Y_test = np.nan_to_num(Y_test)

        return X_test, Y_test
    
    else:
        #=== Finish Extracting training and validation indices ===# 
        X_train = np.empty((0, len(input_variables)))
        Y_train = np.empty((0, len(target_variables)))
        X_valid = np.empty((0, len(input_variables)))
        Y_valid = np.empty((0, len(target_variables)))
    
        for loc in locations:
            # --- training ---#
            X_loc = ERA5[input_variables].sel(time=slice(*dates_range)).sel(time=~validation_times, location=loc).to_array().values.T
            X_train = np.concatenate((X_train, X_loc), axis=0)
            Y_loc = ChSh_Coeff.sel(time=slice(*dates_range)).sel(coeff=target_variables,time=~validation_times, obs=loc).to_array().values
            Y_train = np.concatenate((Y_train, Y_loc[0,:,:]), axis=0)
    
            # --- vlaidation ---#
            X_loc = ERA5[input_variables].sel(time=slice(*dates_range)).sel(time=validation_times, location=loc).to_array().values.T
            X_valid = np.concatenate((X_valid, X_loc), axis=0)
            Y_loc = ChSh_Coeff.sel(time=slice(*dates_range)).sel(coeff=target_variables,time=validation_times, obs=loc).to_array().values
            Y_valid = np.concatenate((Y_valid, Y_loc[0,:,:]), axis=0)        
    
        # Replace NaN values with zeros
        X_train = np.nan_to_num(X_train)
        Y_train = np.nan_to_num(Y_train)
        X_valid = np.nan_to_num(X_valid)
        Y_valid = np.nan_to_num(Y_valid)
        
        return X_train, Y_train, X_valid, Y_valid

# === training and validation data parameters ===#
X_train,Y_train, X_valid,Y_valid = data_processing(input_variables,target_variables,train_dates_range,train_locations)
print('training inputs shape:',X_train.shape,'training targets shape:',Y_train.shape,'validation inputs shape:',X_valid.shape,'validation targets shape:',Y_valid.shape)

# === normalizing the training and validaiton data ---#
min_max_scaler = preprocessing.MinMaxScaler().fit(Y_train)

Y_train_trans = min_max_scaler.transform(Y_train)
Y_valid_trans = min_max_scaler.transform(Y_valid)

def hexbin_plotter(gs,Y,pred,title,text_arg=None):
    errMAE    = mae(Y,pred)
    errRMSE   = np.sqrt(mse(Y,pred))
    errMAPE   = mape(Y,pred)
    errR2     = R2(Y,pred)

    ax_hexbin = fig.add_subplot(gs)
    hb = ax_hexbin.hexbin(np.squeeze(Y), np.squeeze(pred), gridsize=100, bins='log', cmap='inferno')
    if text_arg:
        ax_hexbin.text(0.05, 0.93, f'MAE: {errMAE:.2f} \n$R^2$: {errR2:.2f}\nRMSE: {errRMSE:.2f} \nMAPE: {errMAPE:.2f}',
                      transform=ax_hexbin.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_hexbin.set_xlabel(f'observed coefficient')
    ax_hexbin.set_ylabel(f'Predicted coefficient')
    ax_hexbin.set_title(f'{title}')

    min_value = Y.min()
    max_value = Y.max()
    ax_hexbin.set_xlim(min_value, max_value)
    ax_hexbin.set_ylim(min_value, max_value)

OUTPUT_DIR = f'models/coefficient_{target_variables[0]}/Ens_{Ens}'
os.system(f'mkdir -p {OUTPUT_DIR}')
# --- save the normalizing function ---#
joblib.dump(min_max_scaler, f'{OUTPUT_DIR}/min_max_scaler.joblib')

valMin = 1e8
for i in range(nTrial):
    n_d_opt = np.squeeze(np.random.choice(n_d,1))
    n_a_opt = n_d_opt
    n_steps_opt = np.squeeze(np.random.choice(n_steps,1))
    n_independent_opt = np.squeeze(np.random.choice(n_independent,1))
    n_shared_opt = np.squeeze(np.random.choice(n_shared,1))
    gamma_opt = np.squeeze(np.random.choice(gamma,1))
    print("trial = {:d} n_d = {:d} n_steps = {:d} n_independent = {:d}"
          " n_shared = {:d} gamma = {:f}".format(i,n_d_opt.item(),n_steps_opt.item(),n_independent_opt.item(),n_shared_opt.item(),gamma_opt.item()))

    tabReg   = TabNetRegressor(n_d = n_d_opt.item(), 
                                   n_a = n_a_opt.item(), 
                                   n_steps = n_steps_opt.item(),
                                   n_independent = n_independent_opt.item(),
                                   n_shared = n_shared_opt.item(),
                                   gamma = gamma_opt.item(),
                                   verbose=1,seed=randSeed, )
    tabReg.fit(X_train=X_train, y_train=Y_train_trans,
                      eval_set=[(X_train, Y_train_trans), (X_valid, Y_valid_trans)],
                      eval_name=['train', 'valid'],
                      max_epochs=250, batch_size=512,    #bSize_opt.item(), 
                      eval_metric=['rmse'], patience=10,  #mae, rmse
                      loss_fn = torch.nn.MSELoss())
    
    rmseVal = tabReg.history['valid_rmse'][tabReg.best_epoch]
    if rmseVal < valMin: 
        valMin = rmseVal
        fSTR = f'{OUTPUT_DIR}/TabNet_HOLDOUT_Ens_{str(Ens)}.pkl'
        with open(fSTR, "wb") as f:
            dump(tabReg, f, pickle.HIGHEST_PROTOCOL)
        print('dumped')

    # --- Plot loss curve and hexbin ---
    fig = plt.figure(figsize=(6, 3), constrained_layout=True)
    gs = fig.add_gridspec(1,2)

    # Line plot for train and validation RMSE
    ax = fig.add_subplot(gs[0])
    ax.plot(tabReg.history['train_rmse'], label='train')
    ax.plot(tabReg.history['valid_rmse'], label='validation')
    ax.set_title('Training and Validation RMSE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True)
    
    Y_pred = min_max_scaler.inverse_transform(tabReg.predict(X_valid))
    hexbin_plotter(gs[1],Y_valid[:,0],Y_pred[:,0],f'Coefficient {target_variables[0]}',text_arg=True)
    
    fig.suptitle(f'n_d:{n_d_opt.item()}, n_a: {n_a_opt.item()}, n_steps:{n_steps_opt.item()}, n_independent: {n_independent_opt.item()}, n_shared: {n_shared_opt.item()},gamma:{gamma_opt.item(0)}')
    # Adjust layout
    plt.savefig(f'{OUTPUT_DIR}/n_d{n_d_opt.item()}-n_a{n_a_opt.item()}-n_steps{n_steps_opt.item()}-n_independent{n_independent_opt.item()}-n_shared{n_shared_opt.item()}-gamma{gamma_opt.item()}.png')

