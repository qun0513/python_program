import salem

import xarray as xr
import numpy as np
from xarray.backends import NetCDF4DataStore
import salem
from datetime import datetime
#from siphon.catalog import TDSCatalog
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas

ds=xr.open_dataset('D:/XC/CHN_gridded_pm25_1979_2019_daily.nc')

shp_path = 'D:/XC/China_shp_lqy/'
shp = geopandas.read_file(shp_path + 'province.shp')
criterion = (shp['省']=='广东省')


shp = shp[criterion]
ds =ds.salem.roi(shape=shp)
