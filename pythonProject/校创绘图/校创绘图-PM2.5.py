#!/usr/bin/env python
# coding: utf-8

# In[1]:


#（1980年冬季为1979年12月份，1980年1月份，2月份）
#（2020年冬季为2019年12月份，2020年1月份，2月份）
#这里的PM2.5只有到2019年12月份，对应2020冬季为2019年12月份
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
ds=xr.open_dataset('D:/XC/CHN_gridded_pm25_1979_2019_daily.nc')
lat=ds.lat
lon=ds.lon
#筛选出1979.12--2019.12
data3=ds.isel(time=slice(334,14975))
#41年温度场PM2.5气候态平均
pm3=data3.mean(dim='time')
pm3=pm3.pm25


# In[41]:


#筛选高值年
#(3622,3712) (3987,4077) (4717,4808) (5083,5173) (7639,7730) (10196,10286) (13849,13939) (14944,14975)
data1_1989=(ds.isel(time=slice(3622,3712))).mean(dim='time')
pm_1989=data1_1989.pm25
data1_1990=(ds.isel(time=slice(3987,4077))).mean(dim='time')
pm_1990=data1_1990.pm25
data1_1992=(ds.isel(time=slice(4717,4808))).mean(dim='time')
pm_1992=data1_1992.pm25
data1_1993=(ds.isel(time=slice(5083,5173))).mean(dim='time')
pm_1993=data1_1993.pm25
data1_2000=(ds.isel(time=slice(7639,7730))).mean(dim='time')
pm_2000=data1_2000.pm25
data1_2007=(ds.isel(time=slice(10196,10286))).mean(dim='time')
pm_2007=data1_2007.pm25
data1_2017=(ds.isel(time=slice(13849,13939))).mean(dim='time')
pm_2017=data1_2017.pm25
data1_2020=(ds.isel(time=slice(14944,14975))).mean(dim='time')
pm_2020=data1_2020.pm25
pm1=(pm_1989+pm_1990+pm_1992+pm_1993+pm_2000+pm_2007+pm_2017+pm_2020)/8

#筛选低值年
#(2161,2251) (2526,2616) (6178,6269) (8005,8095)  (9100,9191) (11292,11382) (11657,11747) (12388,12478)
data2_1985=(ds.isel(time=slice(2161,2251))).mean(dim='time')
pm_1985=data2_1985.pm25
data2_1986=(ds.isel(time=slice(2526,2616))).mean(dim='time')
pm_1986=data2_1986.pm25
data2_1996=(ds.isel(time=slice(6178,6269))).mean(dim='time')
pm_1996=data2_1996.pm25
data2_2001=(ds.isel(time=slice(8005,8095))).mean(dim='time')
pm_2001=data2_2001.pm25
data2_2004=(ds.isel(time=slice(9100,9191))).mean(dim='time')
pm_2004=data2_2004.pm25
data2_2010=(ds.isel(time=slice(11292,11382))).mean(dim='time')
pm_2010=data2_2010.pm25
data2_2011=(ds.isel(time=slice(11657,11747))).mean(dim='time')
pm_2011=data2_2011.pm25
data2_2013=(ds.isel(time=slice(12388,12478))).mean(dim='time')
pm_2013=data2_2013.pm25
pm2=(pm_1985+pm_1986+pm_1996+pm_2001+pm_2004+pm_2010+pm_2011+pm_2013)/8


# In[42]:


import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
ax.add_feature(china,lw=0.5,zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'))  #scale:110m\50m\10m
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.tick_params(labelsize=16)
ax.set_title("PM2.5高值年平均值")

pm=pm1
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter =LONGITUDE_FORMATTER
gl.yformatter =LATITUDE_FORMATTER
clevs=np.linspace(-40,150,191)
plt.rcParams['font.sans-serif']=['FangSong'] #用来正常显示中文标签 SimHei
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
contourf=ax.contourf(lon,lat,pm,clevs,cmap=plt.cm.RdBu_r,transform=ccrs.PlateCarree(),extend='both')
plt.colorbar(contourf,ax=ax,fraction=0.03,pad=0.09)
plt.show()