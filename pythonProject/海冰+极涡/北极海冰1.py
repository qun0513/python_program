import xarray as xr
a=xr.open_dataset('D:/GC/HadISST_sst_187001-201903.nc')
print(a.sst)