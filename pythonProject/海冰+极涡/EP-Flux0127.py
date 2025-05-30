import xarray as xr
import numpy as np
import pandas as pd
import matplotlib             #解决Linux无法可视化的问题
matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
from pandas import DataFrame
import time
from datetime import datetime
import netCDF4
import math
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray import DataArray

EPy=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/Daily/epflux/Fy_1979-2018.nc')
EPz=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/Daily/epflux/Fz_1979-2018.nc')
Div=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/Daily/epflux/old/div_1979-2018.nc')


EPy['time']=pd.date_range(start='1/1979',end='1/2019',freq='m')
EPz['time']=pd.date_range(start='1/1979',end='1/2019',freq='m')
Div['time']=pd.date_range(start='1/1979',end='1/2019',freq='m')


#读取数据-------------------------------------------------------------------------
#(epy\epz的分辨率，且加了400、600hPa)
epy0=EPy.Fy.loc[:,0,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epy1=epy.Fy.loc[:,1,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epy2=epy.Fy.loc[:,2,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epy3=epy.Fy.loc[:,3,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epy4=epy.Fy.loc[:,4,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#EPY=(epy0+epy1+epy2+epy3+epy4)/5

epz0=EPz.Fz.loc[:,0,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epz1=epz.Fz.loc[:,1,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epz2=epz.Fz.loc[:,2,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epz3=epz.Fz.loc[:,3,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#epz4=epz.Fz.loc[:,4,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#EPZ=(epz0+epz1+epz2+epz3+epz4)/5

div0=Div.Div.loc[:,0,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-6)]
#div1=div.Div.loc[:,1,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
#div2=div.Div.loc[:,2,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
#div3=div.Div.loc[:,3,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
#div4=div.Div.loc[:,4,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
#DIV=(div0+div1+div2+div3+div4)/5

epy=epy0
epz=epz0
div=div0  #------------------------------------******
epz=100*epz    #30(d_val)   8

lat=epy.lat    #-3-  31      (回改)
level=epy.lev  #37               。。
lat,level=np.meshgrid(lat,level)

#20230127 修改
epy12=epy.sel(time=epy.time.dt.month.isin([12]))
epy1=epy.sel(time=epy.time.dt.month.isin([1]))
epy2010_12=epy12.sel(time=epy12.time.dt.year.isin([2010]))
epy2011_1=epy1.sel(time=epy1.time.dt.year.isin([2010]))
epy_2010w=epy2010_12.mean(dim='time').data/2+epy2011_1.mean(dim='time').data/2

epz12=epy.sel(time=epz.time.dt.month.isin([12]))
epz1=epy.sel(time=epz.time.dt.month.isin([1]))
epz2010_12=epz12.sel(time=epz12.time.dt.year.isin([2010]))
epz2011_1=epz1.sel(time=epz1.time.dt.year.isin([2011]))
epz_2010w=epz2010_12.mean(dim='time').data/2+epz2011_1.mean(dim='time').data/2

div12=epy.sel(time=div.time.dt.month.isin([12]))
div1=epy.sel(time=div.time.dt.month.isin([1]))
div2010_12=div12.sel(time=div12.time.dt.year.isin([2010]))
div2011_1=div1.sel(time=div1.time.dt.year.isin([2011]))
div_2010w=div2010_12.mean(dim='time').data/2+div2011_1.mean(dim='time').data/2
print(epy_2010w.shape,epz_2010w.shape,div_2010w.shape)

div_w12=div.sel(time=div.time.dt.month.isin([12]))
div_w12a=div_w12.sel(time=div_w12.time.dt.year.isin(np.arange(1979,2017)))
div_w1=div.sel(time=div.time.dt.month.isin([1]))
div_w1a=div_w1.sel(time=div_w1.time.dt.year.isin(np.arange(1980,2018)))

div_12=div_w12a.mean(dim='time')
div_1=div_w1a.mean(dim='time')
div_w=div_12/2+div_1/2
div_2010h=div_2010w-div_w


# 放大200hPa以上
def enhance(epy,epz):
  for i in np.arange(0,9):
    for j in np.arange(0,15):
        if i==4:
            epy[i, j] = epy[i, j] *2
            epz[i, j] = epz[i, j] *2  #0,1,2波*2
        if i>=5:
            epy[i,j]=epy[i,j]*3
            epz[i,j]=epz[i,j]*3      #0,1,2波*3

enhance(epy_2010w,epz_2010w)

#绘图------------------------------------------------------------------------------
import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueWhiteOrangeRed#BlueDarkRed18   temp_19lev  BlRe（师兄）
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

fig,ax=plt.subplots()

#lat=div.lat;level=div.lev  #(回改)
#print(lat.size,level.size)
#print(lat,level)     #levels=np.linspace(-2.5e-5,2.5e-5,11),
plt.contourf(lat,level,div_2010w,levels=np.linspace(-1.5e8,1.5e8,11),cmap=cp)#,levels=np.linspace(-2.6e-5,2.6e-5,11)
#,levels=np.linspace(-3.2e-5,3.2e-5,11)
#,levels=np.linspace(-1.2e-4,1.2e-4,9)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.05, aspect=25, shrink=1)
cb.set_label('divergence',size=3,rotation=90,labelpad=5,fontsize=18)
#cb.set_ticks([-1.2e-4,-0.9e-4,-0.6e-4,-0.3e-4,0,0.3e-4,0.6e-4,0.9e-4,1.2e-4])
#[-3.2e-5,-2.56e-5,-1.92e-5,-1.28e-5,-0.64e-5,0,0.64e-5,1.28e-5,1.92e-5,2.56e-5,3.2e-5]
cb.ax.tick_params(size=6,labelsize=16)

#lat=epy.lat;level=epy.lev                          #150000000(0,1,2波)
plt.quiver(lat,level,epy_2010w,epz_2010w, width=0.0025,scale=150000000, color='k')
#bha  bla  oha  ola  bhw  blw  ohw  olw              750000000(1)/300000000(0+hl)
ax.invert_yaxis()
ax.set_yscale('symlog')
ax.set_ylim(700,10)
ax.set_yticks([7e2,5e2,3e2,2e2,1e2,5e1,3e1,2e1,1e1])
ax.set_yticklabels([700,500,300,200,100,50,30,20,10])
ax.set_title('EP-Flux_bw2010',fontsize=18)
ax.tick_params(size=6,labelsize=16)

print('hello,world!')
plt.savefig('/home/dell/ZQ19/EP/ep/EP-Flux_bw2010.jpg')

