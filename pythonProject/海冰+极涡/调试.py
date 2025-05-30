import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset

a0=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')  #bs-barents
#print('a','\n',a)
bs=a0.sic.loc[:,82.5:66.5,15.5:60.5]      #巴伦支海海域
#print('b','\n',b)
bst=bs.isel(time=slice(1319,1802))       #1979-2020 (1979.12-2020.02)
#print('c','\n',c)
b=bst.mean(dim='longitude')              #barents
b=b.mean(dim='latitude')
#print(c[11])

bmon_12=b[0::12]                       #12月原始数据
bmon_1=b[1::12]
bmon_2=b[2::12]

bwinter0=np.empty((41,3))
bwinter0[:,0]=bmon_12
bwinter0[:,1]=bmon_1
bwinter0[:,2]=bmon_2
bwinter1=bwinter0.mean(axis=1)          #冬季三月求平均
#print('winter','\n',winter1)

##################################################################################################
import scipy
from scipy import signal
bmon12=scipy.signal.detrend(bmon_12)    #去趋势
bmon1=scipy.signal.detrend(bmon_1)
bmon2=scipy.signal.detrend(bmon_2)
bwinter=scipy.signal.detrend(bwinter1)
#print(mon12)

#计算标准差
bvwinter=np.var(bwinter)                #方差  variance
bvmon12=np.var(bmon12)
bvmon1=np.var(bmon1)
bvmon2=np.var(bmon2)
bsdwinter=np.sqrt(bvwinter)             #标准差 standard deviation -sd
bsdmon12=np.sqrt(bvmon12)
bsdmon1=np.sqrt(bvmon1)
bsdmon2=np.sqrt(bvmon2)
#print(sdmon12)


a1=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')     #e-Okhotsk
#print('a','\n',a)
es=a1.sic.loc[:,62.5:44.5,135.5:164.5]  #鄂霍茨克海海域
#print('b','\n',b)
est=es.isel(time=slice(1319,1802))       #1979-2020 (1979.12-2020.02)
#print(c)
#print('c','\n',c)
e=est.mean(dim='longitude')
e=e.mean(dim='latitude')
#print(c[11])

emon_12=e[0::12]                       #12月原始数据
emon_1=e[1::12]
emon_2=e[2::12]

ewinter0=np.empty((41,3))
ewinter0[:,0]=emon_12
ewinter0[:,1]=emon_1
ewinter0[:,2]=emon_2
ewinter1=ewinter0.mean(axis=1)
#print('winter','\n',winter)

################################################################################
import scipy
from scipy import signal
emon12=scipy.signal.detrend(emon_12)    #去趋势
emon1=scipy.signal.detrend(emon_1)
emon2=scipy.signal.detrend(emon_2)
ewinter=scipy.signal.detrend(ewinter1)
#print(emon12)

#计算标准差
evwinter=np.var(ewinter)                #方差  variance
evmon12=np.var(emon12)
evmon1=np.var(emon1)
evmon2=np.var(emon2)
esdwinter=np.sqrt(evwinter)             #标准差 standard deviation -sd
esdmon12=np.sqrt(evmon12)
esdmon1=np.sqrt(evmon1)
esdmon2=np.sqrt(evmon2)
#print(esdmon12)


#绘图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)
ax.set_title('1979-2019巴伦支海和鄂霍茨克海海冰去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=15)
#ax.set_xticks(np.arange(1979,2020,2))
#ax.axhline(y=sdmon2,lw=6,ls=':',c='r',label='sd')
#ax.axhline(y=-sdmon2,lw=6,ls=':',c='r')
plt.plot(x,bwinter,lw='10',c='b',label='barents')
plt.plot(x,ewinter,lw='10',c='r',label='okhotsk')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='best',fontsize=60)
plt.show()




'''   两个海域筛选高低值年份，并绘图
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
c=b.isel(time=slice(1319,1802))       #1979-2020 (1979.12-2022.02)
print('c','\n',c)
#c=c.mean(dim='longitude')
#c=c.mean(dim='latitude')

#冬季高值年份筛选
h12=c.sel(time=c.time.dt.month.isin([12]))
sh_12=h12.sel(time=h12.time.dt.year.isin([1979,1982,2000,2011,2018,2019]))
h1=c.sel(time=c.time.dt.month.isin([1]))
sh_1=h1.sel(time=h1.time.dt.year.isin([1980,1983,2001,2012,2019,2020]))
h2=c.sel(time=c.time.dt.month.isin([2]))
sh_2=h2.sel(time=h2.time.dt.year.isin([1980,1983,2001,2012,2019,2020]))
sh12=sh_12.mean(dim='time')
sh1=sh_1.mean(dim='time')
sh2=sh_2.mean(dim='time')
shw=(sh12.data+sh1.data+sh2.data)/3               #数据
lat=sh12.latitude                                 #纬度
lon=sh12.longitude                                #经度
#print('sh12','\n',sh12)
#print('sh1','\n',sh1)
#print('sh2','\n',sh2)
#print('shw','\n',shw)
#print(lat)
#print(lon)

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

import proplot as plot
proj = plot.Proj('lcc', lon_0=38)
fig,axs = plot.subplots(ncols=1,width=10,refwidth=10,height=10,projection=proj,axwidth=10)

axs.format(reso='hi',coast=True,metalinewidth=2,coastlinewidth=0.8,
           lonlim=(14,62),latlim=(65,83),                      #82.5:66.5,15.5:60.5
           title='Lambert Conformal Conic',titlesize=15,titlepad=10,
           )

gl=axs[0].gridlines(
    xlocs=np.arange(-180, 180 + 1, 10), ylocs=np.arange(-90, 90 + 1, 5),
    draw_labels=True, x_inline=False, y_inline=False,
    linewidth=0.5, linestyle='--', color='gray')

gl.top_labels = False
gl.right_labels = False
gl.rotate_labels =0

plt.contourf(lon, lat, shw,levels=np.arange(-0.4,1.4,0.1),cmap=mpl.cm.YlOrBr,extend='both')
#MPL_PiYG_r
#plt.title('2019-01-16',position=(0.5,0.9),loc='center',fontsize=20)

#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=axs[0], orientation="vertical", pad=0.0001, aspect=16, shrink=0.8,drawedges=True)
cb.set_label('',size=14,rotation=0,labelpad=5,fontsize=20)
cb.ax.tick_params(labelsize=10)

plt.show()
'''

'''    筛选年月份
w=[]
x1=0;y1=0;z1=0
for yr in np.arange(1979,1980,1):
  T_year = c.loc[c.time.dt.year.isin([yr])]  # or f.loc
  x=T_year.loc[T_year.time.dt.month.isin([12])]
  y=T_year.loc[T_year.time.dt.month.isin([1])]
  z=T_year.loc[T_year.time.dt.month.isin([2])]
  #x=x.data
  #y=y.data
  #z=z.data
  #print(x)
  w0=(x+y+z)/3
  print(w0)
  #w0=np.array(w0)
  w.append(w0)
  #w=xr.concat(w,dim='time')

#t=np.array(w).reshape(41,3)
print('w','\n',w)
#print('t','\n',t)
#t=t.mean(axis=1)
#print('t','\n',t)
#print('ccc',c.data)
print(x)
'''