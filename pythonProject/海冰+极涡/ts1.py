import xarray as xr
import matplotlib             #解决Linux无法可视化的问题
matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


U=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/U_1979-201908.nc')
u=U.u.loc[:,300,90:40,0:359]
SIC=xr.open_dataset('/home/dell/ZQ19/HadISST_ice_187001-202102.nc')
sic=SIC.sic.loc[:,89.5:39.5,-179.5:179.5]
sic=sic.isel(time=slice(1308,1796))          #1979-2020 (1979.1-2019.08)
Z=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/Z_1979-201908.nc')
z=Z.z.loc[:,300,90:40,0:359]
z=z/10
#print('u','\n',u.data)
#print('z','\n',z.data)
lon=z.longitude
lat=z.latitude
lon=lon[::10]
#lat=lat[::2]

# U   纬向速度
#秋季
year=np.arange(1979,2019)
t1=u.sel(time=u.time.dt.month.isin([9]))
t2=u.sel(time=u.time.dt.month.isin([10]))
t3=u.sel(time=u.time.dt.month.isin([11]))
T1=t1.sel(time=t1.time.dt.year.isin([year]))
T2=t2.sel(time=t2.time.dt.year.isin([year]))
T3=t3.sel(time=t3.time.dt.year.isin([year]))
UA=(T1.data+T2.data+T3.data)/3
#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t4=u.sel(time=u.time.dt.month.isin([12]))
t5=u.sel(time=u.time.dt.month.isin([1]))
t6=u.sel(time=u.time.dt.month.isin([2]))
T4=t4.sel(time=t4.time.dt.year.isin([year1]))
T5=t5.sel(time=t5.time.dt.year.isin([year2]))
T6=t6.sel(time=t6.time.dt.year.isin([year2]))
UW=(T4.data+T5.data+T6.data)/3

ua=np.array(UA).reshape(40,18360)
uw=np.array(UW).reshape(40,18360)
ua=np.transpose(ua)                #矩阵转置
uw=np.transpose(uw)
ua=ua[::10]                        #1836*40
uw=uw[::10]
print(ua.shape)

# SIC
#秋季
year=np.arange(1979,2019)
t1=sic.sel(time=sic.time.dt.month.isin([9]))
t2=sic.sel(time=sic.time.dt.month.isin([10]))
t3=sic.sel(time=sic.time.dt.month.isin([11]))
T1=t1.sel(time=t1.time.dt.year.isin([year]))
T2=t2.sel(time=t2.time.dt.year.isin([year]))
T3=t3.sel(time=t3.time.dt.year.isin([year]))
SA=(T1.data+T2.data+T3.data)/3
#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t4=sic.sel(time=sic.time.dt.month.isin([12]))
t5=sic.sel(time=sic.time.dt.month.isin([1]))
t6=sic.sel(time=sic.time.dt.month.isin([2]))
T4=t4.sel(time=t4.time.dt.year.isin([year1]))
T5=t5.sel(time=t5.time.dt.year.isin([year2]))
T6=t6.sel(time=t6.time.dt.year.isin([year2]))
SW=(T4.data+T5.data+T6.data)/3

#print('T6','\n',T6)
#print(UW.shape)

sa=np.array(SA).reshape(40,18360)
sw=np.array(SW).reshape(40,18360)
sa=np.transpose(sa)                #矩阵转置
sw=np.transpose(sw)
sa=sa[::10]                        #1836*40
sw=sw[::10]
print(sa.shape)

# Z  位势高度场
#秋季
year=np.arange(1979,2019)
t7=z.sel(time=z.time.dt.month.isin([9]))
t8=z.sel(time=z.time.dt.month.isin([10]))
t9=z.sel(time=z.time.dt.month.isin([11]))
T7=t7.sel(time=t7.time.dt.year.isin([year]))
T8=t8.sel(time=t8.time.dt.year.isin([year]))
T9=t9.sel(time=t9.time.dt.year.isin([year]))
ZA=(T7.data+T8.data+T9.data)/3
#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t10=z.sel(time=z.time.dt.month.isin([12]))
t11=z.sel(time=z.time.dt.month.isin([1]))
t12=z.sel(time=z.time.dt.month.isin([2]))
T10=t10.sel(time=t10.time.dt.year.isin([year1]))
T11=t11.sel(time=t11.time.dt.year.isin([year2]))
T12=t12.sel(time=t12.time.dt.year.isin([year2]))
ZW=(T10.data+T11.data+T12.data)/3

#print('T10','\n',T10)

za=np.array(ZA).reshape(40,18360)
zw=np.array(ZW).reshape(40,18360)
za=np.transpose(za)
zw=np.transpose(zw)
za=za[::10]                        #1836*40
zw=zw[::10]
print(za.shape)

'''
import xmca
za=np.transpose(za)
svd=xmca(ua,za)
svd.solver()
lp,rp=svd.patterns(n=2)
le,re=svd.expansionCoefs(n=2)
frac=svd.covFracs(n=2)
print(frac)
'''

