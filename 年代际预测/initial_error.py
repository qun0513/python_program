import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob

def f(year):

    fig=plt.figure(figsize=(25,10),dpi=200)  #每年一张图

    directory = "D:\decadal prediction\data\hindcast\MIROC6"
    file_pattern = f"tos_Omon_MIROC6_dcppA-hindcast_s{year}*.nc"
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    for i in np.arange(0,10):
        