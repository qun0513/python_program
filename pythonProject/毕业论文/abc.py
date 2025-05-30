import numpy as np
import xarray as xr
abc=xr.open_dataset('C:/Users/zhaoqun/Desktop/abc.nc')
print(abc.z)
print(abc)