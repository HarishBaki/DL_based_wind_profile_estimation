# === importing dependencies ===#
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import ast
import yaml
import time

from libraries import *

'''
This script estimates the Chebyshev coefficients for the CERRA wind profile data of 2000 and 2001 years. 
The data is stored in the data/CERRA_height_level folder. The data is stored in the netcdf format. 
The input data is stored in the following format:
    - Dimensions: obs, time, height
    - Variables: data
The output data is stored in the following format:
    - Dimensions: obs, time, coeff
    - Variables: data
'''

ds = xr.open_mfdataset('data/CERRA_height_level/*.nc',
                       combine='nested',
                       concat_dim='time')
Z = ds.heightAboveGround.data
# Number of observations, time steps, and height levels
n_obs, n_time, n_height = ds.data.shape
# Set p value
p = 4
# Initialize Coeff array
Coeff = np.zeros((n_obs, n_time, p+1))
# Iterate over observations and time steps
for i in range(n_obs):
    U = ds.data.isel(obs=i).data.compute()
    stime = time.time()
    for t in range(n_time):
        Coeff[i, t, :] = Chebyshev_Coeff(Z,U[t, :], p, CPtype)
    etime = time.time()
    print(f'Time elapsed for obs: {i}: {etime-stime}s')
coeff_da = xr.DataArray(Coeff, dims=['obs','time', 'coeff'], coords={'obs':ds.obs,'time': ds.time, 'coeff': np.arange(p+1)},name='data')
coeff_da.to_netcdf('data/Chebyshev_Coefficnents.nc')