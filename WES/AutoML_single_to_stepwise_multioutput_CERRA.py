# %%
from datetime import datetime

# print date as date accessed
date_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Date accessed: {date_accessed}")

import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, re, time, math, calendar

import flaml
from flaml import AutoML
from multiprocessing import Process

import pickle
from pickle import dump, load
import joblib

if os.path.exists('/media/ssd_2tb_evo/DL_based_wind_profile_estimation'):
    root_dir = '/media/ssd_2tb_evo/DL_based_wind_profile_estimation'
elif os.path.exists('/home/harish/Ongoing_Research/DL_based_wind_profile_estimation'):
    root_dir = '/home/harish/Ongoing_Research/DL_based_wind_profile_estimation'
else:
    root_dir = '/Users/harishbaki/Ongoing_Research/DL_based_wind_profile_estimation'

sys.path.append(root_dir)
from libraries import *
from plotters import *

#For reproducibility of the results, the following seeds should be selected 
from numpy.random import seed
randSeed = np.random.randint(1000)

# %%
# === gather variables provided as input arguments ===
config_file = 'config_ERA5_to_CERRA.yaml' #sys.argv[1]
train_years = int(17) #int(sys.argv[2])
Ens = 0 #int(sys.argv[3]) #ensemble number
warm_start = False #bool(sys.argv[4]) #True if you want to warm start the model training
transformed = 'not_transformed' #sys.argv[5] #True if you want to apply the min-max scaler to the target variables

# === load yaml file and extract variables ===
with open(config_file, 'r') as yaml_file:
    configure = yaml.safe_load(yaml_file)
input_file = configure['input_file']
input_times_freq = configure['input_times_freq']
Coeff_file = configure['Coeff_file']
profiles_file = None if configure['profiles_file'] == '' else configure['profiles_file']
input_variables = configure['input_variables']
train_locations = configure['train_locations']
test_dates_range = configure['test_dates_range']
test_locations = configure['test_locations']
nEns = configure['nEns']
tabnet_param_file = configure['tabnet_param_file']
target_variables = configure['target_variables']
experiment = configure['experiment']

model_output_dir = f'trained_models/{experiment}/FLAML/{train_years}/Ens{Ens}'
os.system(f'mkdir -p {model_output_dir}')

# %%
# === Input parameters ===
input_variables = [
    "10ws", "100ws", "100alpha", "975ws", "950ws", "975wsgrad", "950wsgrad",
    "zust", "i10fg", "t2m", "skt", "stl1", "d2m", "msl", "blh", "ishf", 
    "ie", "tcc", "lcc", "cape", "bld", "t_975", "t_950", "2mtempgrad", 
    "sktempgrad", "dewtempsprd", "975tempgrad", "950tempgrad", "sinHR", 
    "cosHR", "sinJDAY", "cosJDAY",
    "u10","v10","u100","v100","u_950","v_950","u_975","v_975",
    "u10_mean","u10_std","u10_skew","u10_kurt",
    "v10_mean","v10_std","v10_skew","v10_kurt",
    "u100_mean","u100_std","u100_skew","u100_kurt",
    "v100_mean","v100_std","v100_skew","v100_kurt",
    "u_950_mean","u_950_std","u_950_skew","u_950_kurt",
    "v_950_mean","v_950_std","v_950_skew","v_950_kurt",
    "u_975_mean","u_975_std","u_975_skew","u_975_kurt",
    "v_975_mean","v_975_std","v_975_skew","v_975_kurt",
]

#for run,year in enumerate(np.arange(2000,2017+1-train_years)):
train_dates_range = ['2000-01-01T12','2016-12-31']
print(train_dates_range)

# === training and validation data parameters ===#
X_train,Y_train, X_valid,Y_valid = data_processing(input_file,Coeff_file,input_times_freq,
                                                    input_variables,target_variables,train_dates_range,train_locations,val_arg=True,
                                                    profiles_file=profiles_file,threshold=2)
print('training inputs shape:',X_train.shape,'training targets shape:',Y_train.shape,'validation inputs shape:',X_valid.shape,'validation targets shape:',Y_valid.shape)

# === testing data parameters ===#
X_test,Y_test = data_processing(input_file,Coeff_file,input_times_freq,input_variables,target_variables,test_dates_range,test_locations)
print('testing inputs shape:',X_test.shape,'testing targets shape:',Y_test.shape)

# Print the shapes of the final datasets
print(f"Final X_train shape: {X_train.shape}")
print(f"Final Y_train shape: {Y_train.shape}")
print(f"Final X_valid shape: {X_valid.shape}")
print(f"Final Y_valid shape: {Y_valid.shape}")
print(f"Final X_test shape: {X_test.shape}")
print(f"Final Y_test shape: {Y_test.shape}")

# %%
# === Train the model ===
# === normalizing the training and validaiton data ---#
if transformed == 'transformed':
    min_max_scaler = preprocessing.MinMaxScaler().fit(Y_train)

    Y_train = min_max_scaler.transform(Y_train)
    Y_valid = min_max_scaler.transform(Y_valid)

    # --- save the normalizing function ---#
    joblib.dump(min_max_scaler, f'{model_output_dir}/min_max_scaler.joblib')
    print('min_max_scaler dumped')

automl_settings = {
    "time_budget": 3600,  # in seconds
    "metric": 'rmse',
    "task": 'regression',
    "estimator_list": ['xgboost'],
    "early_stop": True,
    "model_history": True, #A boolean of whether to keep the best model per estimator
    "retrain_full": True, #whether to retrain the selected model on the full training data
    "custom_hp": {
        "xgboost": {
            "tree_method": {
                "domain": "gpu_hist",       # Use GPU for tree construction
                "type": "fixed"
            },
            "predictor": {
                "domain": "gpu_predictor",  # Use GPU for prediction
                "type": "fixed"
            }
        }
    }
}

# === running the 9 process across 3 GPUs ===
gpu_devices = [0, 1, 2]

# Function to train base models
def train_base_model(target_variable, gpu_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print(f"Training base model for target {target_variable} on GPU {gpu_device}")

    automl = AutoML()
    X_tr, y_tr = X_train, Y_train[:, target_variable:target_variable+1]
    X_val, y_val = X_valid, Y_valid[:, target_variable:target_variable+1]

    model_path = f'{model_output_dir}/C{target_variable}.pkl'
    # --- for warm start ---#
    if warm_start and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            automl_prev_model = load(f)
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,starting_points=automl_prev_model.best_config_per_estimator, **automl_settings)
    else:
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val, **automl_settings)
    with open(model_path, "wb") as f:
        dump(automl, f)
    print(f"Base model {target_variable} saved to {model_path}")

# Function to train stepping stone models
def train_step_model(target_variable, gpu_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print(f"Training stepping stone model for target {target_variable} on GPU {gpu_device}")

    automl = AutoML()
    X_tr = np.hstack([X_train, Y_train[:, :target_variable]])
    y_tr = Y_train[:, target_variable:target_variable+1]
    X_val = np.hstack([X_valid, Y_valid[:, :target_variable]])
    y_val = Y_valid[:, target_variable:target_variable+1]

    model_path = f'{model_output_dir}/C{target_variable}_step{target_variable}.pkl'
    # --- for warm start ---#
    if warm_start and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            automl_prev_model = load(f)
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,starting_points=automl_prev_model.best_config_per_estimator, **automl_settings)
    else:
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val, **automl_settings)
    with open(model_path, "wb") as f:
        dump(automl, f)
    print(f"Step model {target_variable} saved to {model_path}")

# Launch parallel training for base models
processes = []
idx = 0
for target_variable in ([0,1,2,3,4]):
    gpu_device = gpu_devices[idx % len(gpu_devices)]
    p = Process(target=train_base_model, args=(target_variable, gpu_device))
    p.start()
    processes.append(p)
    idx += 1

# Launch parallel training for stepping stone models
for target_variable in ([1, 2, 3, 4]):
    gpu_device = gpu_devices[idx % len(gpu_devices)]
    p = Process(target=train_step_model, args=(target_variable, gpu_device))
    p.start()
    processes.append(p)
    idx += 1

for p in processes:
    p.join()

print("Training completed.")

# %%
# === Plotting hexbins ===
fig = plt.figure(figsize=(15, 13), constrained_layout=True)
gs = fig.add_gridspec(4,len(target_variables))

# --- First row, with step 0 ---
Y_pred = []
for target_variable in target_variables:
    # load the respective model
    fSTR = f'{model_output_dir}/C{target_variable}.pkl'
    with open(fSTR, "rb") as f:
        model = load(f)
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1,1)
    Y_pred = np.hstack([Y_pred,y_pred]) if target_variable>0 else y_pred

if transformed == 'transformed':
    min_max_scaler = joblib.load(f'{model_output_dir}/min_max_scaler.joblib')
    Y_pred = min_max_scaler.inverse_transform(Y_pred)

for target_variable in (target_variables):
    ylabel = 'Single target\n Predicted' if target_variable == 0 else ''
    hexbin_plotter(fig,gs[0,target_variable],Y_test[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel=ylabel)

for target_variable in (target_variables):
    ylabel = 'Single target\n Predicted' if target_variable == 0 else ''
    ax_qq = fig.add_subplot(gs[1,target_variable])
    QQ_plotter(ax_qq,Y_test[:,target_variable],Y_pred[:,target_variable],title=f'Coefficient {target_variable}',label='',color='blue',xlabel='True',ylabel=ylabel,one_to_one=True)

# --- Second row, with step 1 to 4 ---
Y_pred = []
for target_variable in ([0,1,2,3,4]):
    # load the respective model
    fSTR = f'{model_output_dir}/C{target_variable}.pkl' if target_variable==0 else f'{model_output_dir}/C{target_variable}_step{target_variable}.pkl'
    with open(fSTR, "rb") as f:
        model = load(f)
    X_te = X_test if target_variable==0 else np.hstack([X_test,Y_pred])
    y_te = Y_test[:,target_variable:target_variable+1]
    y_pred = model.predict(X_te)
    y_pred = y_pred.reshape(-1,1)
    Y_pred = np.hstack([Y_pred,y_pred]) if target_variable>0 else y_pred

if transformed == 'transformed':
    min_max_scaler = joblib.load(f'{model_output_dir}/min_max_scaler.joblib')
    Y_pred = min_max_scaler.inverse_transform(Y_pred)

for target_variable in ([1,2,3,4]):
    ylabel = 'Steppingwise target\n Predicted' if target_variable == 1 else ''
    hexbin_plotter(fig,gs[2,target_variable],Y_test[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel=ylabel)

for target_variable in ([1,2,3,4]):
    ax_qq = fig.add_subplot(gs[3,target_variable])
    ylabel = 'Stepwise target\n Predicted' if target_variable == 0 else ''
    QQ_plotter(ax_qq,Y_test[:,target_variable],Y_pred[:,target_variable],title=f'Coefficient {target_variable}',label='',color='blue',xlabel='True',ylabel=ylabel,one_to_one=True)
plt.savefig(f'{model_output_dir}/hexbin_{warm_start}.png')
plt.close()