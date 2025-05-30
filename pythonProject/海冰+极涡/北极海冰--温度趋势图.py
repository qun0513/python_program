import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
a=xr.open_dataset('D:/GC/HadISST_sst_187001-201903.nc')
print(a)
print(a.sst)
sst=a.sst
sst=sst.mean(dim='longitude')
sst=sst.mean(dim='latitude')
sst.isel(time=slice(1308,1789))
x=np.arange(1979,2020,480)
fig,ax=plt.subplots()
ax.plot(x,sst)
plt.show()