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
config_file = 'config_ERA5.yaml'

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

# === data processing ===
X_train,Y_train = data_processing_Heligoland(input_file,Coeff_file,input_variables,target_variables,train_times,train_locations)
X_valid,Y_valid = data_processing_Heligoland(input_file,Coeff_file,input_variables,target_variables,valid_times,train_locations)
X_test,Y_test = data_processing_Heligoland(input_file,Coeff_file,input_variables,target_variables,test_times,train_locations)
print('Training inputs shape:',X_train.shape,'training targets shape:',Y_train.shape)
print('Validation inputs shape:',X_valid.shape,'validation targets shape:',Y_valid.shape)
print('Testing inputs shape:',X_test.shape,'testing targets shape:',Y_test.shape)

pretrained_experiment = '17Y'
experiment = 'Heligoland_retraining'
pretrained_run = 0
run = 0
Ens = 0
pretrained_OUTPUT_DIR = f'{root_dir}/WES/trained_models/{pretrained_experiment}/run_{pretrained_run}/Ens_{Ens}'
OUTPUT_DIR = f'{root_dir}/WES/trained_models/{experiment}/run_{run}/Ens_{Ens}'
os.system(f'mkdir -p {OUTPUT_DIR}')
# Load model
fSTR = f'TabNet_HOLDOUT_Ens_{str(Ens)}.pkl'
with open(f'{pretrained_OUTPUT_DIR}/{fSTR}', "rb") as f:
    tabReg = pickle.load(f)
    print(tabReg)