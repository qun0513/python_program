import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
import pandas as pd
from pandas import DataFrame
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

a=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
print('a','\n',a)
b=a.sic.loc[:,82.5:66.5,15.5:60.5]    #巴伦支海海域
print('b','\n',b)
c=b.isel(time=slice(1319,1802))       #1979-2020 (1979.12-2020.02)
print('c','\n',c)
#c=c.mean(dim='longitude')
#c=c.mean(dim='latitude')

#冬季高值年份筛选  h-high
h12=c.sel(time=c.time.dt.month.isin([12]))
sh_12=h12.sel(time=h12.time.dt.year.isin([1997,1998,2002,2003,2010,2014,2019]))
h1=c.sel(time=c.time.dt.month.isin([1]))
sh_1=h1.sel(time=h1.time.dt.year.isin([1998,1999,2003,2004,2011,2014,2018]))
h2=c.sel(time=c.time.dt.month.isin([2]))
sh_2=h2.sel(time=h2.time.dt.year.isin([1998,1999,2003,2004,2011,2014,2018]))
sh12=sh_12.mean(dim='time')
sh1=sh_1.mean(dim='time')
sh2=sh_2.mean(dim='time')
shw=(sh12.data+sh1.data+sh2.data)/3            #数据
lat=c.latitude                                 #纬度
lon=c.longitude                                #经度
#print('sh12','\n',sh12)
#print('sh1','\n',sh1)
#print('sh2','\n',sh2)
#print('shw','\n',shw)
#print(lat)
#print(lon)

#冬季低值年份筛选  l-low
l12=c.sel(time=c.time.dt.month.isin([12]))
sl_12=l12.sel(time=l12.time.dt.year.isin([1982,1984,2005,2007,2015,2016,2017]))
l1=c.sel(time=c.time.dt.month.isin([1]))
sl_1=l1.sel(time=l1.time.dt.year.isin([1983,1985,2006,2008,2016,2017,2018]))
l2=c.sel(time=c.time.dt.month.isin([2]))
sl_2=l2.sel(time=l2.time.dt.year.isin([1983,1985,2006,2008,2016,2017,2018]))
sl12=sl_12.mean(dim='time')
sl1=sl_1.mean(dim='time')
sl2=sl_2.mean(dim='time')
slw=(sl12.data+sl1.data+sl2.data)/3               #数据

#秋季高值年筛选  h-high
h901=c.sel(time=c.time.dt.month.isin([9,10,11]))
sh_901=h901.sel(time=h901.time.dt.year.isin([1982,1988,1993,1998,2002,2003,2014,2019]))
sh901=sh_901.mean(dim='time')
sha=sh901.data
#print(sha)
#秋季低值年筛选  l-low
l901=c.sel(time=c.time.dt.month.isin([9,10,11]))
sl_901=l901.sel(time=h901.time.dt.year.isin([1979,1984,2007,2012,2013]))
sl901=sl_901.mean(dim='time')
sla=sl901.data
#print(sla)

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#兰伯特投影
import proplot as plot
proj = plot.Proj('lcc', lon_0=38)
fig,axs = plot.subplots(ncols=1,width=10,height=10,projection=proj,facecolor='white')

axs.format(reso='hi',coast=True,metalinewidth=2,coastlinewidth=0.8,
           lonlim=(14,62),latlim=(65,83),                      #82.5:66.5,15.5:60.5
           title='Barents sea sea ice\ndifference map in winter',
           titlesize=15,titlepad=10,facecolor='white'
           )
axs.tick_params(labelsize=20)
#Barents Sea sea ice difference map in winter
#Barents Sea sea ice difference map in autumn
gl=axs[0].gridlines(
    xlocs=np.arange(-180, 180 + 1, 10), ylocs=np.arange(-90, 90 + 1, 5),
    draw_labels=True, x_inline=False, y_inline=False,
    linewidth=0.5, linestyle='--', color='none')

gl.top_labels = False
gl.right_labels = False
gl.rotate_labels =0
gl.xlabel_style = {'size':13, 'color':'black'}
gl.ylabel_style = {'size':13, 'color':'black'}

#dif=shw-slw
#print(dif)

plt.contourf(lon, lat, slw-shw,levels=np.arange(-0.7,0.1,0.1),cmap=mpl.cm.YlOrBr_r,extend='both')
#MPL_PiYG_r
#plt.title('2019-01-16',position=(0.5,0.9),loc='center',fontsize=20)

#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=axs[0], orientation="vertical", pad=0.0001, aspect=16, shrink=0.8,drawedges=True)
cb.set_label('',size=14,rotation=0,labelpad=5,fontsize=20)
cb.ax.tick_params(labelsize=10)

plt.show()