import xarray as xr
#import matplotlib             #解决Linux无法可视化的问题
#matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray import DataArray

#数据读取————————————————————————————————————————————————————————————————————————————————————————
#U=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/U_1979-201908.nc')
U=xr.open_dataset('D:/GC/U_1979-201908.nc')
u=U.u.loc[:,500,90:40,0:359]
#SIC=xr.open_dataset('/home/dell/ZQ19/HadISST_ice_187001-202102.nc')
SIC=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
sic=SIC.sic.loc[:,89.5:39.5,-179.5:179.5]

print('u','\n',u,'\n')
print('sic','\n',sic,'\n')

#经度转换----------------------------
import xarray as xr
def wrap_lon_to_180(data: xr.DataArray | xr.Dataset,
                    lon: str = 'lon',
                    center_on_180: bool = True
                    ) -> xr.DataArray | xr.Dataset:
    '''
    Wrap longitude coordinates of DataArray or Dataset to either -180..179 or 0..359.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        An xarray DataArray or Dataset object containing longitude coordinates.
    lon : str, optional
        The name of the longitude coordinate, default is 'lon'.
    center_on_180 : bool, optional
        If True, wrap longitude from 0..359 to -180..179;
        If False, wrap longitude from -180..179 to 0..359.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The DataArray or Dataset with wrapped longitude coordinates.
    '''
    # Wrap -180..179 to 0..359
    if center_on_180:
        data = data.assign_coords(**{lon: (lambda x: (x[lon] % 360))})
    # Wrap 0..359 to -180..179
    else:
        data = data.assign_coords(**{lon: (lambda x: ((x[lon] + 180) % 360) - 180)})
    return data.sortby(lon, ascending=True)

data=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
sic1 = wrap_lon_to_180(data, lon = 'longitude', center_on_180 = True)
print(sic1)

data=xr.open_dataset('D:/GC/U_1979-201908.nc')
u1 = wrap_lon_to_180(data, lon = 'longitude', center_on_180 = False)
print(u1)

print('u1','\n',u1,'\n')
print('sic1','\n',sic1,'\n')


