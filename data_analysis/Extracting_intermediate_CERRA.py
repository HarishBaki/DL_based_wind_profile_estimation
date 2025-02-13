import numpy as np
import pandas as pd
import xarray as xr
import dask
import os
import glob
import sys
import ast
import json

file = sys.argv[1]
lev_type = sys.argv[2]
if lev_type == 'SFC':
    isobaricInhPa = False
else:
    isobaricInhPa = [975,950]
var_names = ast.literal_eval(sys.argv[3])
i = sys.argv[4]

print(file,lev_type,isobaricInhPa,var_names,i)

lats=[54.0143,
        51.97,
        56.44050,
        51.706898,
        51.6463,
        51.9171,
        51.9990,
        52.306583,
        53.71250,
        54,
        54.177256]
lons=[6.58385,
        4.93,
        8.15080,
        3.034618,
        2.95141,
        3.6670,
        3.2760,
        4.008972,
        7.1522,
        6.623333,
        7.891971
        ]

if lev_type == 'SFC':
    datasets = {}
    for var_name in var_names:
        ds = xr.open_dataset(file, engine='cfgrib',backend_kwargs={'indexpath': '','filter_by_keys': {'cfVarName': var_name}})
        datasets[var_name] = ds
    ds = xr.merge(datasets.values(),compat='override')
else:
    ds = xr.open_dataset(file, engine='cfgrib',backend_kwargs={'indexpath': ''})
data = ds[var_names]
result_list_locwise = []
for lat,lon in zip(lats,lons):
    distance_squared = (data.latitude - lat)**2 + (data.longitude - lon)**2
    indices = np.unravel_index(np.nanargmin(distance_squared), distance_squared.shape)
    nearest_indices = {'y': indices[0], 'x': indices[1]}
    result = data.sel(y=nearest_indices['y'],x = nearest_indices['x'])
    if isobaricInhPa:
        result = result.sel(isobaricInhPa=isobaricInhPa)
    result_list_locwise.append(result)
result = xr.concat(result_list_locwise,dim='location')

result.to_netcdf(f'/media/harish/SSD/wind_profile_estimation/data/CERRA_complete/temp_dir/{i}.nc')