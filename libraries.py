
# === importing dependencies ===#
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import ast
import yaml
import time

from pytorch_tabnet.tab_model import TabNetRegressor
import pickle
from pickle import dump, load
import joblib

import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as R2
from scipy.interpolate import interp1d

# The following height levels are fixed at all time
Z = np.array([10., 15., 30., 50., 75., 100., 150., 200., 250., 300., 400., 500.])

# These variables are used while creating the Chebyshev coefficients
poly_order = 4
CPtype = 1

def normalize(H,ref_H=Z):
    '''
    Normalizes the height levels between -1 and 1
    ref_H: A vector of reference height levels, in our case CERRA levels
    H: A vector of height levels
    '''
    a = 2 / (np.max(ref_H) - np.min(ref_H))
    b = - (np.max(ref_H) + np.min(ref_H)) / (np.max(ref_H) - np.min(ref_H))
    Hn = a * ref_H + b

    Hn = np.interp(H, ref_H, Hn)
    return Hn

def Chebyshev_Basu(x, poly_order, CPtype):
    '''
    This function computes the Chebyshev polynomials, according to the equations in Mason, J. C., & Handscomb, D. C. (2002). Chebyshev polynomials. Chapman and Hall/CRC.
    x: input variable, between -1 and 1
    poly_order: order of polynomials
    CPtype: 1 or 2, according to the publication
    '''
    if x.ndim == 1:
        x = x[:, np.newaxis]

    CP = np.zeros((len(x), poly_order + 1))

    CP[:, 0] = 1  # T0(x) = 1
    if poly_order >= 1:
        if CPtype == 1:  # Chebyshev polynomial of first kind
            CP[:, 1] = x.flatten()  # T1(x) = x
        else:  # Chebyshev polynomial of second kind
            CP[:, 1] = 2 * x.flatten()  # U1(x) = 2x
        if poly_order >= 2:
            for n in range(2, poly_order + 1):
                CP[:, n] = 2 * x.flatten() * CP[:, n - 1] - CP[:, n - 2]
    return CP

def Chebyshev_Coeff(H, U,p,CPtype,ref_H=None):
    '''
    This function computes the Chebyshev coefficients through inverse transform of system of linear equations
    H: height levels, in their useual units
    U: wind speed at the height levels
    p: polynomial order
    CPtype: 1 or 2, according to the publication
    '''
    H = H.flatten()
    U = U.flatten()

    # Normalize H
    Hn = normalize(H)
    
    # Remove NaN values
    Indx = np.where(~np.isnan(U))[0]
    Ha = Hn[Indx]
    Ua = U[Indx]
    N = len(Ua)

    # Linearly extrapolate wind values at the boundaries
    spline_left = interp1d(Ha, Ua, kind='linear', fill_value='extrapolate')
    Uax = spline_left([-1])

    spline_right = interp1d(Ha, Ua, kind='linear', fill_value='extrapolate')
    Uay = spline_right([1])
    Ua = np.concatenate([Uax, Ua, Uay])
    Ha = np.concatenate([-1 + np.zeros(1), Ha, 1 + np.zeros(1)])
    
    # Predict the gap-filled and denoised profile
    PL = Chebyshev_Basu(Ha, p, CPtype)
    # Compute the coefficients C
    Coeff = np.linalg.pinv(PL) @ Ua
    return Coeff

def WindProfile(Z,Coeff,ref_H=None):
    '''
    This function computes the full level wind profile provided vertical levels and the Chebyshev coefficients
    Z: height levels, in their useual units
    Coeff: Chebysev coefficients
    '''
    # Normalize H
    Hn = normalize(Z)
    PL_full = Chebyshev_Basu(Hn, poly_order, CPtype)
    Mp = PL_full @ Coeff
    return Mp

def Basu_WindProfile(z, a,b,c):
    '''
    This function computes the full level wind profile provided vertical levels and the Basu coefficients
    Z: height levels, in their useual units
    a,b,c: Basu coefficients
    '''
    return 1 / (a + b * np.log(z) + c * (np.log(z))**2)

def Basu_Coeff(z,u):
    from scipy.optimize import curve_fit
    # Perform curve fitting
    popt, pcov = curve_fit(Basu_WindProfile, z, u, p0=[1, 1, 1])  # p0 are initial guesses for parameters a, b, and c
    return popt

def data_processing(input_file,ChSh_Coeff_file,input_times_freq,input_variables,target_variables, dates_range, locations,val_arg=None,profiles_file=None,threshold=1):
    '''
    This function reads the nc files and converts them into numpy arrays in the required shape.
    input_file: input variables file (either ERA5 or CERRA)
    ChSh_Coeff_file: target variables file (Chebyshev coefficients file)
    input_time_freq: time frequency of the input variables, since the CERRA and ERA5 are not at the same time frequencey
    input_variables: names of the input variables
    target_variables: target Chebyshev coefficients 
    dates_range: range of the dates to read, can be training or testing
    locations: location indices of the data, out of 11
    var_arg: whether the function should return validation data also along with training, or only training, or testing
    '''
    inputs = xr.open_dataset(input_file)
    ChSh_Coeff = xr.open_dataset(ChSh_Coeff_file)
    if profiles_file is not None:
        profiles = xr.open_dataset(profiles_file)

    if val_arg:
        #=== Extracting training and validation indices ===# 
        time_coord = inputs.sel(time=slice(*dates_range)).coords['time']
        years = time_coord.dt.year
        months = time_coord.dt.month
        validation_times = np.zeros(len(time_coord), dtype=bool)
        for year in np.unique(years):
            for month in range(1, 13):
                month_indices = np.where((years == year) & (months == month))[0]
                if len(month_indices) >= int(6*24/input_times_freq):
                    start_index = np.random.choice(len(month_indices) - int(6*24/input_times_freq) - 1)
                    validation_indices = month_indices[start_index:start_index + int(6*24/input_times_freq)]
                    validation_times[validation_indices] = True
                else:
                    # Skip the month if not enough data is available
                    continue
        
        #=== Finish Extracting training and validation indices ===# 
        X_train = np.empty((0, len(input_variables)))
        Y_train = np.empty((0, len(target_variables)))
        X_valid = np.empty((0, len(input_variables)))
        Y_valid = np.empty((0, len(target_variables)))
    
        for loc in locations:
            # --- training ---#
            X_loc = inputs[input_variables].sel(time=slice(*dates_range)).sel(time=~validation_times, location=loc).to_array().values.T
            Y_loc = ChSh_Coeff.sel(time=slice(*dates_range,input_times_freq)).sel(coeff=target_variables,time=~validation_times, obs=loc).to_array().values[0,...]
            
            # --- Exclude profiles below threshold ---#
            if profiles_file is not None:
                indices_below_threshold = np.where(np.sqrt(
                np.mean(
                    (
                        inputs[['10ws','100ws']]
                        .sel(time=slice(*dates_range))
                        .sel(time=~validation_times, location=loc)
                        .to_array()
                        .values.T - profiles.data
                        .sel(time=slice(*dates_range))
                        .sel(time=~validation_times, obs=loc, heightAboveGround=[10.0, 100.])
                        .values
                    )**2,
                    axis=1
                )
                )< threshold)[0]
                X_loc = X_loc[indices_below_threshold]
                Y_loc = Y_loc[indices_below_threshold]
            X_train = np.concatenate((X_train, X_loc), axis=0)
            Y_train = np.concatenate((Y_train, Y_loc), axis=0)
    
            # --- vlaidation ---#
            X_loc = inputs[input_variables].sel(time=slice(*dates_range)).sel(time=validation_times, location=loc).to_array().values.T
            Y_loc = ChSh_Coeff.sel(time=slice(*dates_range,input_times_freq)).sel(coeff=target_variables,time=validation_times, obs=loc).to_array().values[0,...]
            # --- Exclude profiles below threshold ---#
            if profiles_file is not None:
                indices_below_threshold = np.where(np.sqrt(
                np.mean(
                    (
                        inputs[['10ws','100ws']]
                        .sel(time=slice(*dates_range))
                        .sel(time=validation_times, location=loc)
                        .to_array()
                        .values.T - profiles.data
                        .sel(time=slice(*dates_range))
                        .sel(time=validation_times, obs=loc, heightAboveGround=[10.0, 100.])
                        .values
                    )**2,
                    axis=1
                )
                )<= threshold)[0]
                X_loc = X_loc[indices_below_threshold]
                Y_loc = Y_loc[indices_below_threshold]
            X_valid = np.concatenate((X_valid, X_loc), axis=0)
            Y_valid = np.concatenate((Y_valid, Y_loc), axis=0)        
    
        # Replace NaN values with zeros
        X_train = np.nan_to_num(X_train)
        Y_train = np.nan_to_num(Y_train)
        X_valid = np.nan_to_num(X_valid)
        Y_valid = np.nan_to_num(Y_valid)
        
        return X_train, Y_train, X_valid, Y_valid

    else:
        X = np.empty((0, len(input_variables)))
        Y = np.empty((0, len(target_variables)))

        for loc in locations:
            # --- testing ---#
            X_loc = inputs[input_variables].sel(time=slice(*dates_range)).sel(location=loc).to_array().values.T
            X = np.concatenate((X, X_loc), axis=0)
            Y_loc = ChSh_Coeff.sel(time=slice(*dates_range,input_times_freq)).sel(coeff=target_variables, obs=loc).to_array().values
            Y = np.concatenate((Y, Y_loc[0,:,:]), axis=0)

        # Replace NaN values with zeros
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)

        return X, Y

def data_processing_Heligoland(input_file,Coeff_file,input_times_freq,input_variables,target_variables, dates_range, locations,val_arg=None,profiles_file=None,threshold=1,CERRA_coeff_file=None):
    '''
    This function reads the nc files and converts them into numpy arrays in the required shape.
    input_file: input variables file (either ERA5 or CERRA)
    ChSh_Coeff_file: target variables file (Chebyshev coefficients file)
    input_time_freq: time frequency of the input variables, since the CERRA and ERA5 are not at the same time frequencey
    input_variables: names of the input variables
    target_variables: target Chebyshev coefficients 
    dates_range: range of the dates to read, can be training or testing
    locations: location indices of the data, out of 11
    var_arg: whether the function should return validation data also along with training, or only training, or testing
    '''
    inputs = xr.open_dataset(input_file)
    coeff = xr.open_dataset(Coeff_file)

    if val_arg:
        #=== Extracting training and validation indices ===# 
        time_coord = coeff.sel(time=slice(*dates_range)).coords['time'] # this is necessary since the time dimension is not the same in the input and target files

        years = time_coord.dt.year
        months = time_coord.dt.month
        validation_times = np.zeros(len(time_coord), dtype=bool)
        for year in np.unique(years):
            for month in range(1, 13):
                # check if you have enough data in the month
                month_indices = np.where((years == year) & (months == month))[0]
                if len(month_indices) >= int(6*24/input_times_freq):
                    start_index = np.random.choice(len(month_indices) - int(6*24/input_times_freq) - 1)
                    validation_indices = month_indices[start_index:start_index + int(6*24/input_times_freq)]
                    validation_times[validation_indices] = True
                else:
                    # Skip the month if not enough data is available
                    continue
        
        #=== Finish Extracting training and validation indices ===#
        X_train = np.empty((0, len(input_variables)))
        Y_train = np.empty((0, len(target_variables)))
        X_valid = np.empty((0, len(input_variables)))
        Y_valid = np.empty((0, len(target_variables)))

        for loc in locations:
            # --- training ---#
            X_loc = inputs[input_variables].sel(time=time_coord).sel(time=~validation_times,location=loc).to_array().values.T
            Y_loc = coeff.sel(time=time_coord).sel(time=~validation_times,coeff=target_variables).to_array().values[0,...]
            X_train = np.concatenate((X_train, X_loc), axis=0)
            Y_train = np.concatenate((Y_train, Y_loc), axis=0)

            # --- vlaidation ---#
            X_loc = inputs[input_variables].sel(time=time_coord).sel(time=validation_times,location=loc).to_array().values.T
            Y_loc = coeff.sel(time=time_coord).sel(time=validation_times,coeff=target_variables).to_array().values[0,...]
            X_valid = np.concatenate((X_valid, X_loc), axis=0)
            Y_valid = np.concatenate((Y_valid, Y_loc), axis=0)
            
        return X_train, Y_train, X_valid, Y_valid
    else:
        time_coord = coeff.sel(time=slice(*dates_range)).coords['time'] # this is necessary since the time dimension is not the same in the input and target files

        X = np.empty((0, len(input_variables)))
        Y = np.empty((0, len(target_variables)))
        if CERRA_coeff_file is not None:
            CERRA_coeff = xr.open_dataset(CERRA_coeff_file)
            Y_CERRA = np.empty((0, len(target_variables)))

        for loc in locations:
            # --- testing ---#
            X_loc = inputs[input_variables].sel(time=time_coord).sel(location=loc).to_array().values.T
            X = np.concatenate((X, X_loc), axis=0)
            Y_loc = coeff.sel(time=time_coord).sel(coeff=target_variables).to_array().values[0,...]
            Y = np.concatenate((Y, Y_loc), axis=0)
            if CERRA_coeff_file is not None:
                Y_loc_CERRA = CERRA_coeff.sel(time=time_coord).sel(coeff=target_variables,obs=loc).to_array().values[0,...]
                Y_CERRA = np.concatenate((Y_CERRA, Y_loc_CERRA), axis=0)

        # Replace NaN values with zeros
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)

        if CERRA_coeff_file is not None:
            Y_CERRA = np.nan_to_num(Y_CERRA)
            return X, Y, Y_CERRA
        else:
            return X, Y


nELI5max = 1 #FIXME
def myELI5(model,X,y,multiout=None,target_variable=None):
    '''
    Calculates the feature importance using the ELI5 methodology
    model: saved ML model
    X: Input matrix
    y: target vector, since the feature importane is designed for one target variable at a time
    multiout: whether the model is trained with multioutput mode
    target_variable: if the model is trained with multioutput mode, we need to specify the target variable from the prediction
    '''
    nSamples, nFeatures = np.shape(X)
    iTot = np.arange(0,nSamples,1)
    
    #Original prediction
    if multiout:
        y_pred_org = model.predict(X)[:,target_variable]
    else:
        y_pred_org = model.predict(X)
    E_org      = np.sqrt(mse(y,y_pred_org)) 
    
    featImp = np.zeros(nFeatures)
    for nF in range(nFeatures):

        E_shfl_tot = 0
        for nELI5 in range(nELI5max):
            
            X_shfl = np.copy(X)

            np.random.shuffle(iTot)

            dum          = X_shfl[:,nF]
            X_shfl[:,nF] = dum[iTot]
            X_shfl       = pd.DataFrame(data=X_shfl)
            if multiout:
                y_pred_shfl  = model.predict(X_shfl.values)[:,target_variable]
            else:
                y_pred_shfl  = model.predict(X_shfl.values)
            E_shfl       = np.sqrt(mse(y,y_pred_shfl))

            E_shfl_tot   = E_shfl_tot + E_shfl

        #print(nF,E_org,E_shfl_tot/nELI5/E_org)
        featImp[nF] = (E_shfl_tot/nELI5max - E_org)*100/E_org
        
    return featImp


def featImp_variables(target_variable,number_of_features):
    '''
    In training, where certain important features are only used, which are computed before using XGBoost and saved.
    target_variable: target coefficient
    number_of_features: number of important features corresponding the target variable
    '''
    # === Load important features ===#
    sorted_feature_importance_array = np.load(f'Coefficient_{target_variable}_featImp.npy')    
    # Access the data from the numpy array
    feature_names = sorted_feature_importance_array['Feature']
    importances = sorted_feature_importance_array['Importance']
    return feature_names[:number_of_features[target_variable]]

