import numpy as np
import pandas as pd
import xarray as xr
import dask
import os
import glob
import sys

target_file = sys.argv[1]

files = sorted(glob.glob(f'CERRA_complete/temp_dir/*.nc'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
xr.open_mfdataset(files,concat_dim='valid_time',parallel=True,combine='nested').to_netcdf(target_file)
