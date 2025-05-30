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

ep0=xr.open_dataset('/home/cesm/tmp/ep/ep_ep_0.monthmcat.nc')
print(ep0)
print(ep0.level,"xxxxxxxxxxxxxxxxxxxxxxxxxxx")
epy=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/Daily/epflux/Fy_1979-2018.nc')
epz=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/Daily/epflux/Fz_1979-2018.nc')
div=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/Daily/epflux/old/div_1979-2018.nc')
print(epy)
#print(epy.time)
#print('div','\n',div,'\n')

#epy= epy.assign_coords(time = epy.indexes['time'].to_datetimeindex())
#epy.time= datetime.strptime(epy.time, "%Y%m%d")
#epy.time=netCDF4.num2date(epy.time,epy.time.units)
#epy.time= datetime.fromtimestamp(epy.time)
#epy.time=time.localtime(epy.time)
#epy.time=time.strftime("%Y-%m-%d",epy.time)

#epy['time']=epy['time'].astype(str)
#epy['time']=pd.to_datetime(epy['time'])

#epy.assign_coords(time1=pd.to_datetime(epy.time.values,infer_datetime_format=True))
#epy['time'] = pd.DatetimeIndex(epy['time'].values)
#epz['time'] = pd.DatetimeIndex(epz['time'].values)
#div['time'] = pd.DatetimeIndex(div['time'].values)
#epy['time']=epy['time'].dt.date
#epz['time']=epz['time'].dt.date
#div['time']=div['time'].dt.date

epy['time']=pd.date_range(start='1/1979',end='1/2019',freq='m')
epz['time']=pd.date_range(start='1/1979',end='1/2019',freq='m')
div['time']=pd.date_range(start='1/1979',end='1/2019',freq='m')

ep0['time']=pd.date_range(start='1/1979',end='9/2019',freq='m')



#print(epy,'\n','xxxxxxxxxxxxxxxxxxxxxxxx')
#print(epy.time)
#dt64为datetime64类型的变量

#读取数据-------------------------------------------------------------------------
#(epy\epz的分辨率，且加了400、600hPa)
epy0=epy.Fy.loc[:,0,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
epy1=epy.Fy.loc[:,1,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
epy2=epy.Fy.loc[:,2,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
epy3=epy.Fy.loc[:,3,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
epy4=epy.Fy.loc[:,4,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
EPY=(epy0+epy1+epy2+epy3+epy4)/5
#epy=epy0
#print(epy.lev,epy.lat,'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')

epz0=epz.Fz.loc[:,0,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
epz1=epz.Fz.loc[:,1,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
epz2=epz.Fz.loc[:,2,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
epz3=epz.Fz.loc[:,3,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
epz4=epz.Fz.loc[:,4,[700,500,300,200,100,50,30,20,10],np.arange(90,0,-6)]
EPZ=(epz0+epz1+epz2+epz3+epz4)/5
#epz=epz0
#print(epz.lev,epz.lat,'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')

div0=div.Div.loc[:,0,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
div1=div.Div.loc[:,1,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
div2=div.Div.loc[:,2,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
div3=div.Div.loc[:,3,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
div4=div.Div.loc[:,4,[700,600,500,400,300,200,100,50,30,20,10],np.arange(90,0,-1)]
DIV=(div0+div1+div2+div3+div4)/5
#epy=epy0
#print(div.lev,div.lat,'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
#print(div0)

#epy0=ep0.Fphi.loc[:,[10.,20.,30.,50.,100.,200.,300.,400.,500.,600.,700.],np.arange(90,0,-1)]
#epz0=ep0.Fz.loc[:,[10.,20.,30.,50.,100.,200.,300.,400.,500.,600.,700.],np.arange(90,0,-1)]
#div0=ep0.Dudt.loc[:,[10.,20.,30.,50.,100.,200.,300.,400.,500.,600.,700.],np.arange(90,0,-1)]

epy1979_1=ep0.Fphi.loc[:,[10.,20.,30.,50.,100.,200.,300.,400.,500.,600.,700.],np.arange(90,0,-6)]
epz1979_1=ep0.Fz.loc[:,[10.,20.,30.,50.,100.,200.,300.,400.,500.,600.,700.],np.arange(90,0,-6)]
div1979_1=ep0.Dudt.loc[:,[10.,20.,30.,50.,100.,200.,300.,400.,500.,600.,700.],np.arange(90,0,-6)]

epy1979_1=epy1979_1.isel(time=0)
epz1979_1=100*epz1979_1.isel(time=0)
div1979_1=div1979_1.isel(time=0)

epy=epy1979_1
epz=epz1979_1
div=div1979_1  #------------------------------------******
epz=100*epz    #30(d_val)   8

'''
div_bar=div.mean(dim='time')

div_a=div.sel(time=div.time.dt.month.isin([9,10,11]))
div_aa=div_a.sel(time=div_a.time.dt.year.isin(np.arange(1979,2018)))
div_a_bar=div_aa.mean(dim='time')

div_w1=div.sel(time=div.time.dt.month.isin([12]))
div_wa1=div_w1.sel(time=div_w1.time.dt.year.isin(np.arange(1979,2017)))
div_w2=div.sel(time=div.time.dt.month.isin([1,2]))
div_wa2=div_w2.sel(time=div_w2.time.dt.year.isin(np.arange(1980,2018)))
div_w21=div.sel(time=div.time.dt.month.isin([1]))
div_wa21=div_w21.sel(time=div_w21.time.dt.year.isin(np.arange(1980,2018)))

div_wa1_bar=div_wa1.mean(dim='time')
div_wa2_bar=div_wa2.mean(dim='time')
div_wa21_bar=div_wa21.mean(dim='time')
div_w_bar=div_wa1_bar/3+div_wa2_bar*2/3
div_w_bar1=div_wa1_bar/2+div_wa21_bar/2


lat=epy.latitude    #-3-  31      (回改)
level=epy.level  #37               。。
lat,level=np.meshgrid(lat,level)

#year=np.arange(1979,2020)
#t=c.sel(time=c.time.dt.month.isin([9,10,11]))
#T=t.sel(time=t.time.dt.year.isin([year]))
#print(level,'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

#秋季-------------------------------------------------------------------
year_bsha=[1982,1988,1993,1998,2002,2003,2014]       #7
year_bsla=[1979,1984,2007,2012,2013]                 #5
year_osha=[1979,1980,1998,2013,2014,2015,2016]       #7
year_osla=[1991,1995,1997,1999,2003,2005,2007,2008]  #8
#epy------------------------------(不同海域高低值)
mon0=epy.sel(time=epy.time.dt.month.isin([9,10,11]))
epy_bsha=mon0.sel(time=mon0.time.dt.year.isin([year_bsha]))  #
epy_bha=epy_bsha.mean(dim='time')
mon1=epy.sel(time=epy.time.dt.month.isin([9,10,11]))
epy_bsla=mon1.sel(time=mon1.time.dt.year.isin([year_bsla]))  #
epy_bla=epy_bsla.mean(dim='time')
mon2=epy.sel(time=epy.time.dt.month.isin([9,10,11]))
epy_osha=mon2.sel(time=mon2.time.dt.year.isin([year_osha]))  #
epy_oha=epy_osha.mean(dim='time')
mon3=epy.sel(time=epy.time.dt.month.isin([9,10,11]))
epy_osla=mon3.sel(time=mon3.time.dt.year.isin([year_osla]))  #
epy_ola=epy_osla.mean(dim='time')
#epz------------------------------（..）
mon4=epz.sel(time=epz.time.dt.month.isin([9,10,11]))
epz_bsha=mon4.sel(time=mon4.time.dt.year.isin([year_bsha]))  #
epz_bha=epz_bsha.mean(dim='time')
mon5=epz.sel(time=epz.time.dt.month.isin([9,10,11]))
epz_bsla=mon5.sel(time=mon5.time.dt.year.isin([year_bsla]))  #
epz_bla=epz_bsla.mean(dim='time')
mon6=epz.sel(time=epz.time.dt.month.isin([9,10,11]))
epz_osha=mon6.sel(time=mon6.time.dt.year.isin([year_osha]))  #
epz_oha=epz_osha.mean(dim='time')
mon7=epz.sel(time=epz.time.dt.month.isin([9,10,11]))
epz_osla=mon7.sel(time=mon7.time.dt.year.isin([year_osla]))  #
epz_ola=epz_osla.mean(dim='time')
#div------------------------------(..)
mon8=div.sel(time=div.time.dt.month.isin([9,10,11]))
div_bsha=mon8.sel(time=mon8.time.dt.year.isin([year_bsha]))
div_bha=div_bsha.mean(dim='time')
mon9=div.sel(time=div.time.dt.month.isin([9,10,11]))
div_bsla=mon9.sel(time=mon9.time.dt.year.isin([year_bsla]))
div_bla=div_bsla.mean(dim='time')
mon10=div.sel(time=div.time.dt.month.isin([9,10,11]))
div_osha=mon10.sel(time=mon10.time.dt.year.isin([year_osha]))
div_oha=div_osha.mean(dim='time')
mon11=div.sel(time=div.time.dt.month.isin([9,10,11]))
div_osla=mon11.sel(time=mon11.time.dt.year.isin([year_osla]))
div_ola=div_osla.mean(dim='time')

#冬季---------------------------------------------------------------------
year_bshw=[1997,1998,2002,2003,2010,2014]           #6
year_bsh_w=[1998,1998,2003,2004,2011,2015]
year_bslw=[1982,1984,2005,2007,2015,2016,2017]      #7
year_bsl_w=[1983,1985,2006,2008,2016,2017,2018]
year_oshw=[1979,1982,2000,2011]                     #4
year_osh_w=[1980,1983,2001,2012]
year_oslw=[1983,1990,1995,2005,2008]                #5
year_osl_w=[1984,1991,1996,2006,2009]
#epy------------------------------------(..)
mon0=epy.sel(time=epy.time.dt.month.isin([12]))
mon_0=epy.sel(time=epy.time.dt.month.isin([1,2]))
mon_01=epy.sel(time=epy.time.dt.month.isin([1]))

epy_bshw=mon0.sel(time=mon0.time.dt.year.isin([year_bshw]))
epy_bsh_w=mon_0.sel(time=mon_0.time.dt.year.isin([year_bsh_w]))
epy_bsh_w1=mon_01.sel(time=mon_01.time.dt.year.isin([year_bsh_w]))
#print(epy_bshw)
T0=epy_bshw.mean(dim='time')
T_0=epy_bsh_w.mean(dim='time')
T_01=epy_bsh_w1.mean(dim='time')
epy_bhw=T0.data/3+T_0.data*2/3  #
epy_bhw1=T0.data/2+T_01.data/2

#mon0=epy.sel(time=epy.time.dt.month.isin([9,10,11]))
#epy_bsha=mon0.sel(time=mon0.time.dt.year.isin([year_bsha]))  #

mon1=epy.sel(time=epy.time.dt.month.isin([12]))
mon_1=epy.sel(time=epy.time.dt.month.isin([1,2]))
mon_11=epy.sel(time=epy.time.dt.month.isin([1]))
epy_bslw=mon1.sel(time=mon1.time.dt.year.isin([year_bslw]))
epy_bsl_w=mon_1.sel(time=mon_1.time.dt.year.isin([year_bsl_w]))
epy_bsl_w1=mon_11.sel(time=mon_11.time.dt.year.isin([year_bsl_w]))
T1=epy_bslw.mean(dim='time')
T_1=epy_bsl_w.mean(dim='time')
T_11=epy_bsl_w1.mean(dim='time')
epy_blw=T1.data/3+T_1.data*2/3  #
epy_blw1=T1.data/2+T_11.data/2  #

mon2=epy.sel(time=epy.time.dt.month.isin([12]))
mon_2=epy.sel(time=epy.time.dt.month.isin([1,2]))
mon_21=epy.sel(time=epy.time.dt.month.isin([1]))
epy_oshw=mon2.sel(time=mon2.time.dt.year.isin([year_oshw]))
epy_osh_w=mon_2.sel(time=mon_2.time.dt.year.isin([year_osh_w]))
epy_osh_w1=mon_21.sel(time=mon_21.time.dt.year.isin([year_osh_w]))
T2=epy_oshw.mean(dim='time')
T_2=epy_osh_w.mean(dim='time')
T_21=epy_osh_w1.mean(dim='time')
epy_ohw=T2.data/3+T_2.data*2/3  #
epy_ohw1=T2.data/2+T_21.data/2  #

mon3=epy.sel(time=epy.time.dt.month.isin([12]))
mon_3=epy.sel(time=epy.time.dt.month.isin([1,2]))
mon_31=epy.sel(time=epy.time.dt.month.isin([1]))
epy_oslw=mon3.sel(time=mon3.time.dt.year.isin([year_oslw]))
epy_osl_w=mon_3.sel(time=mon_3.time.dt.year.isin([year_osl_w]))
epy_osl_w1=mon_31.sel(time=mon_31.time.dt.year.isin([year_osl_w]))
T3=epy_oslw.mean(dim='time')
T_3=epy_osl_w.mean(dim='time')
T_31=epy_osl_w1.mean(dim='time')
epy_olw=T3.data/3+T_3.data*2/3  #
epy_olw1=T3.data/2+T_31.data/2  #
#epz------------------------------------(..)
mon0=epz.sel(time=epz.time.dt.month.isin([12]))
mon_0=epz.sel(time=epz.time.dt.month.isin([1,2]))
mon_01=epz.sel(time=epz.time.dt.month.isin([1]))

epz_bshw=mon0.sel(time=mon0.time.dt.year.isin([year_bshw]))
epz_bsh_w=mon_0.sel(time=mon_0.time.dt.year.isin([year_bsh_w]))
epz_bsh_w1=mon_01.sel(time=mon_01.time.dt.year.isin([year_bsh_w]))
T0=epz_bshw.mean(dim='time')
T_0=epz_bsh_w.mean(dim='time')
T_01=epz_bsh_w1.mean(dim='time')
epz_bhw=T0.data/3+T_0.data*2/3  #
epz_bhw1=T0.data/2+T_01.data/2

mon1=epz.sel(time=epz.time.dt.month.isin([12]))
mon_1=epz.sel(time=epz.time.dt.month.isin([1,2]))
mon_11=epz.sel(time=epz.time.dt.month.isin([1]))
epz_bslw=mon1.sel(time=mon1.time.dt.year.isin([year_bslw]))
epz_bsl_w=mon_1.sel(time=mon_1.time.dt.year.isin([year_bsl_w]))
epz_bsl_w1=mon_11.sel(time=mon_11.time.dt.year.isin([year_bsl_w]))
T1=epz_bslw.mean(dim='time')
T_1=epz_bsl_w.mean(dim='time')
T_11=epz_bsl_w1.mean(dim='time')
epz_blw=T1.data/3+T_1.data*2/3  #
epz_blw1=T1.data/2+T_11.data/2  #

mon2=epz.sel(time=epz.time.dt.month.isin([12]))
mon_2=epz.sel(time=epz.time.dt.month.isin([1,2]))
mon_21=epz.sel(time=epz.time.dt.month.isin([1]))
epz_oshw=mon2.sel(time=mon2.time.dt.year.isin([year_oshw]))
epz_osh_w=mon_2.sel(time=mon_2.time.dt.year.isin([year_osh_w]))
epz_osh_w1=mon_21.sel(time=mon_21.time.dt.year.isin([year_osh_w]))
T2=epz_oshw.mean(dim='time')
T_2=epz_osh_w.mean(dim='time')
T_21=epz_osh_w1.mean(dim='time')
epz_ohw=T2.data/3+T_2.data*2/3  #
epz_ohw1=T2.data/2+T_21.data/2  #

mon3=epz.sel(time=epz.time.dt.month.isin([12]))
mon_3=epz.sel(time=epz.time.dt.month.isin([1,2]))
mon_31=epz.sel(time=epz.time.dt.month.isin([1]))
epz_oslw=mon3.sel(time=mon3.time.dt.year.isin([year_oslw]))
epz_osl_w=mon_3.sel(time=mon_3.time.dt.year.isin([year_osl_w]))
epz_osl_w1=mon_31.sel(time=mon_31.time.dt.year.isin([year_osl_w]))
T3=epz_oslw.mean(dim='time')
T_3=epz_osl_w.mean(dim='time')
T_31=epz_osl_w1.mean(dim='time')
epz_olw=T3.data/3+T_3.data*2/3  #
epz_olw1=T3.data/2+T_31.data/2  #

#div--------------------------------------(..)
mon0=div.sel(time=div.time.dt.month.isin([12]))
mon_0=div.sel(time=div.time.dt.month.isin([1,2]))
mon_01=div.sel(time=div.time.dt.month.isin([1]))
div_bshw=mon0.sel(time=mon0.time.dt.year.isin([year_bshw]))
div_bsh_w=mon_0.sel(time=mon_0.time.dt.year.isin([year_bsh_w]))
div_bsh_w1=mon_01.sel(time=mon_01.time.dt.year.isin([year_bsh_w]))
T0=div_bshw.mean(dim='time')
T_0=div_bsh_w.mean(dim='time')
T_01=div_bsh_w1.mean(dim='time')
div_bhw=T0.data/3+T_0.data*2/3#
div_bhw1=T0.data/2+T_01.data/2

mon1=div.sel(time=div.time.dt.month.isin([12]))
mon_1=div.sel(time=div.time.dt.month.isin([1,2]))
mon_11=div.sel(time=div.time.dt.month.isin([1]))
div_bslw=mon1.sel(time=mon1.time.dt.year.isin([year_bslw]))
div_bsl_w=mon_1.sel(time=mon_1.time.dt.year.isin([year_bsl_w]))
div_bsl_w1=mon_11.sel(time=mon_11.time.dt.year.isin([year_bsl_w]))
T1=div_bslw.mean(dim='time')
T_1=div_bsl_w.mean(dim='time')
T_11=div_bsl_w1.mean(dim='time')
div_blw=T1.data/3+T_1.data*2/3#
div_blw1=T1.data/2+T_11.data/2#

mon2=div.sel(time=div.time.dt.month.isin([12]))
mon_2=div.sel(time=div.time.dt.month.isin([1,2]))
mon_21=div.sel(time=div.time.dt.month.isin([1]))
div_oshw=mon2.sel(time=mon2.time.dt.year.isin([year_oshw]))
div_osh_w=mon_2.sel(time=mon_2.time.dt.year.isin([year_osh_w]))
div_osh_w1=mon_21.sel(time=mon_21.time.dt.year.isin([year_osh_w]))
T2=div_oshw.mean(dim='time')
T_2=div_osh_w.mean(dim='time')
T_21=div_osh_w1.mean(dim='time')
div_ohw=T2.data/3+T_2.data*2/3#
div_ohw1=T2.data/2+T_21.data/2#

mon3=div.sel(time=div.time.dt.month.isin([12]))
mon_3=div.sel(time=div.time.dt.month.isin([1,2]))
mon_31=div.sel(time=div.time.dt.month.isin([1]))
div_oslw=mon3.sel(time=mon3.time.dt.year.isin([year_oslw]))
div_osl_w=mon_3.sel(time=mon_3.time.dt.year.isin([year_osl_w]))
div_osl_w1=mon_31.sel(time=mon_31.time.dt.year.isin([year_osl_w]))
T3=div_oslw.mean(dim='time')
T_3=div_osl_w.mean(dim='time')
T_31=div_osl_w1.mean(dim='time')
div_olw=T3.data/3+T_3.data*2/3#
div_olw1=T3.data/2+T_31.data/2#


epz_bsha=7*epz_bsha
epz_bsla=7*epz_bsla
epz_osha=7*epz_osha
epz_osla=7*epz_osla
epz_bshw=7*epz_bshw
epz_bslw=7*epz_bslw
epz_oshw=7*epz_oshw
epz_oslw=7*epz_oslw
print(epy_bsha.shape,epz_bsha.shape)


# 放大200hPa以上
def enhance(epy,epz):
  for i in np.arange(0,9):
    for j in np.arange(0,15):
        if i==4:
            epy[i, j] = epy[i, j] *6
            epz[i, j] = epz[i, j] *6  #0,1,2波*2
        if i>=5:
            epy[i,j]=epy[i,j]*12
            epz[i,j]=epz[i,j]*12      #0,1,2波*3

#秋季----------------------------------------------------------------------
enhance(epy_bha,epz_bha)  #(不同海域高低值)
enhance(epy_bla,epz_bla)
enhance(epy_oha,epz_oha)
enhance(epy_ola,epz_ola)
#冬季----------------------------------------------------------------------
enhance(epy_bhw1,epz_bhw1)
enhance(epy_blw1,epz_blw1)
enhance(epy_ohw1,epz_ohw1)
enhance(epy_olw1,epz_olw1)


#散度显著性检验(单样本t检验)---------------------------------------
#准备数据
#秋季--------------------------------------------------------------------------
  #巴伦支海--------------------
t01=div.sel(time=div.time.dt.month.isin([9]))
t02=div.sel(time=div.time.dt.month.isin([10]))
t03=div.sel(time=div.time.dt.month.isin([11]))
T01=t01.sel(time=t01.time.dt.year.isin([year_bsha]))    #high
T02=t02.sel(time=t02.time.dt.year.isin([year_bsha]))
T03=t03.sel(time=t03.time.dt.year.isin([year_bsha]))
T11=t01.sel(time=t01.time.dt.year.isin([year_bsla]))    #low
T12=t02.sel(time=t02.time.dt.year.isin([year_bsla]))
T13=t03.sel(time=t03.time.dt.year.isin([year_bsla]))

div_b_ha=(T01.data+T02.data+T03.data)/3   #high(7)
div_b_la=(T11.data+T12.data+T13.data)/3   #low (8)

dif1=np.empty((7,11,90))
dif2=np.empty((5,11,90))
s=0
for k in np.arange(0,7):
    dif1[k,:,:]=(div_b_ha[k,:,:]-div_a_bar)   #海冰增多的影响
    s=s+dif1[k,:,:]
dif1_bar=s/7
s=0
for k in np.arange(0,5):
    dif2[k,:,:]=(div_b_la[k,:,:]-div_a_bar)   #海冰减少的影响
    s=s+dif2[k,:,:]
dif2_bar=s/5

  #鄂霍茨克海-------------------
t11=div.sel(time=div.time.dt.month.isin([9]))
t12=div.sel(time=div.time.dt.month.isin([10]))
t13=div.sel(time=div.time.dt.month.isin([11]))
T21=t01.sel(time=t11.time.dt.year.isin([year_osha]))    #high
T22=t02.sel(time=t12.time.dt.year.isin([year_osha]))
T23=t03.sel(time=t13.time.dt.year.isin([year_osha]))
T31=t01.sel(time=t11.time.dt.year.isin([year_osla]))    #low
T32=t02.sel(time=t12.time.dt.year.isin([year_osla]))
T33=t03.sel(time=t13.time.dt.year.isin([year_osla]))

div_o_ha=(T21.data+T22.data+T23.data)/3   #high(7)  --差值显著性检验
div_o_la=(T31.data+T32.data+T33.data)/3   #low (8)  ——差值..

dif3=np.empty((7,11,90))
dif4=np.empty((8,11,90))
s=0
for k in np.arange(0,7):
    dif3[k,:,:]=(div_o_ha[k,:,:]-div_a_bar)    #高值年显著性检验
    s = s + dif3[k, :, :]
dif3_bar = s / 7
s=0
for k in np.arange(0,8):
    dif4[k,:,:]=(div_o_la[k,:,:]-div_a_bar)    #低值年显著性检验
    s = s + dif4[k, :, :]
dif4_bar = s / 8

#冬季--------------------------------------------------------------------
  #巴伦支海--------------------
t21=div.sel(time=div.time.dt.month.isin([12]))
t22=div.sel(time=div.time.dt.month.isin([1]))
t23=div.sel(time=div.time.dt.month.isin([2]))
T41=t21.sel(time=t21.time.dt.year.isin([year_bshw]))    #high
T42=t22.sel(time=t22.time.dt.year.isin([year_bshw]))
T43=t23.sel(time=t23.time.dt.year.isin([year_bshw]))
T51=t21.sel(time=t21.time.dt.year.isin([year_bslw]))    #low
T52=t22.sel(time=t22.time.dt.year.isin([year_bslw]))
T53=t23.sel(time=t23.time.dt.year.isin([year_bslw]))

div_b_hw=(T41.data+T42.data+T43.data)/3   #high(6)  --差值显著性检验
div_b_lw=(T51.data+T52.data+T53.data)/3   #low (7)  ——差值..

div_b_hw1=(T41.data+T42.data)/2   #high(6)  --差值显著性检验
div_b_lw1=(T51.data+T52.data)/2   #low (7)  ——差值..

dif5=np.empty((6,11,90))
dif6=np.empty((7,11,90))
s=0
for k in np.arange(0,6):
    dif5[k,:,:]=(div_b_hw1[k,:,:]-div_w_bar1)    #高值年显著性检验
    s = s + dif5[k, :, :]
dif5_bar = s / 6
s=0
for k in np.arange(0,7):
    dif6[k,:,:]=(div_b_lw1[k,:,:]-div_w_bar1)    #低值年显著性检验
    s = s + dif6[k, :, :]
dif6_bar = s / 7

  #鄂霍茨克海-----------------
t31=div.sel(time=div.time.dt.month.isin([12]))
t32=div.sel(time=div.time.dt.month.isin([1]))
t33=div.sel(time=div.time.dt.month.isin([2]))
T61=t31.sel(time=t31.time.dt.year.isin([year_oshw]))    #high
T62=t32.sel(time=t32.time.dt.year.isin([year_oshw]))
T63=t33.sel(time=t33.time.dt.year.isin([year_oshw]))
T71=t31.sel(time=t31.time.dt.year.isin([year_oslw]))    #low
T72=t32.sel(time=t32.time.dt.year.isin([year_oslw]))
T73=t33.sel(time=t33.time.dt.year.isin([year_oslw]))

div_o_hw=(T41.data+T42.data+T43.data)/3   #high(6)  --差值显著性检验
div_o_lw=(T51.data+T52.data+T53.data)/3   #low (7)  ——差值..

dif7=np.empty((4,11,90))
dif8=np.empty((5,11,90))
s=0
for k in np.arange(0,4):
    dif7[k,:,:]=(div_o_hw[k,:,:]-div_w_bar)    #高值年显著性检验
    s = s + dif7[k, :, :]
dif7_bar = s / 4
s=0
for k in np.arange(0,5):
    dif8[k,:,:]=(div_o_lw[k,:,:]-div_w_bar)    #低值年显著性检验
    s = s + dif8[k, :, :]
dif8_bar = s / 5


from scipy.stats import ttest_1samp
t1,p1=ttest_1samp(dif1,0)
t2,p2=ttest_1samp(dif2,0)
t3,p3=ttest_1samp(dif3,0)
t4,p4=ttest_1samp(dif4,0)
t5,p5=ttest_1samp(dif5,0)
t6,p6=ttest_1samp(dif6,0)
t7,p7=ttest_1samp(dif7,0)
t8,p8=ttest_1samp(dif8,0)
#差值dif1_bar

#双样本t检验-------------------------
from scipy import stats
from scipy.stats import ttest_ind
t11,p11=ttest_ind(div_b_ha,div_b_la,equal_var=False)
t22,p22=ttest_ind(div_b_hw,div_b_lw,equal_var=False)
t33,p33=ttest_ind(div_o_ha,div_o_la,equal_var=False)
t44,p44=ttest_ind(div_o_hw,div_o_lw,equal_var=False)

#散度差值
dif11=div_bla-div_bha                                 #巴伦支海秋季
dif22=div_ola-div_oha
dif33=div_blw1-div_bhw1
dif44=div_olw1-div_ohw1

#dif55=epz_blw1-epz_bhw1


#ep通量差值
epy11=epy_bla-epy_bha
epy22=epy_ola-epy_oha
epy33=epy_blw1-epy_bhw1
epy44=epy_olw1-epy_ohw1

epz11=epz_bla-epz_bha
epz22=epz_ola-epz_oha
epz33=epz_blw1-epz_bhw1
epz44=epz_olw1-epz_ohw1


#掩膜掉不打点的区域
import numpy.ma as ma
p1=ma.MaskedArray(data = p2, mask = np.logical_and(p1<1,p1>0.05))

#plt.rcParams['font.sans-serif']=['KaiTi'] #用来正常显示中文标签 SimHei FangSong
#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
'''
#绘图------------------------------------------------------------------------------
import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlRe#BlueDarkRed18   temp_19lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

fig,ax=plt.subplots()

lat=div.latitude;level=div.level  #(回改)
#print(lat,level)     #levels=np.linspace(-2.5e-5,2.5e-5,11),
plt.contourf(lat,level,div1979_1,levels=np.linspace(-15,15,21),cmap=cp)#,levels=np.linspace(-2.6e-5,2.6e-5,11)
#,levels=np.linspace(-3.2e-5,3.2e-5,11)
#,levels=np.linspace(-1.2e-4,1.2e-4,9)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.05, aspect=25, shrink=1)
cb.set_label('divergence',size=3,rotation=90,labelpad=5,fontsize=18)
#cb.set_ticks([-1.2e-4,-0.9e-4,-0.6e-4,-0.3e-4,0,0.3e-4,0.6e-4,0.9e-4,1.2e-4])
#[-3.2e-5,-2.56e-5,-1.92e-5,-1.28e-5,-0.64e-5,0,0.64e-5,1.28e-5,1.92e-5,2.56e-5,3.2e-5]
cb.ax.tick_params(size=6,labelsize=16)
#print(epy,epz)
#print(lat,level)

#lat=epy.lat;level=epy.lev                           #150000000(0,1,2波)
plt.quiver(lat,level,epy1979_1,epz1979_1, width=0.0025,scale=1500, color='k')
#bha  bla  oha  ola  bhw  blw  ohw  olw              750000000(1)/300000000(0+hl)
ax.invert_yaxis()
ax.set_yscale('symlog')
ax.set_ylim(700,10)
ax.set_yticks([7e2,5e2,3e2,2e2,1e2,5e1,3e1,2e1,1e1])
ax.set_yticklabels([700,500,300,200,100,50,30,20,10])
ax.set_title('EP-Flux_111',fontsize=18)
ax.tick_params(size=6,labelsize=16)

#打点---------------------------------------------
#pp=ax.scatter(xx[::7],yy[::7],marker='.',s=2.8,c='k',alpha=1) #-----------

#lat=div.lat;level=div.lev
#pp=ax.contourf(lat,level,p1,hatches=['.....'],colors="none",zorder=0)

#plt.rcParams['font.sans-serif']=['KaiTi'] #用来正常显示中文标签 SimHei FangSong
#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

print('hello,world!')
plt.savefig('/home/dell/ZQ19/EP/ep1/EP-Flux_111.jpg')


''''
dif1_bar + p1
dif2_bar + p2
dif11 + p11
'''

'''
epy0=epy.Fy.loc[:,0,::2,np.arange(90,0,-3)]
epy1=epy.Fy.loc[:,1,::2,np.arange(90,0,-3)]
epy2=epy.Fy.loc[:,2,::2,np.arange(90,0,-3)]
epy3=epy.Fy.loc[:,3,::2,np.arange(90,0,-3)]
epy4=epy.Fy.loc[:,4,::2,np.arange(90,0,-3)]
epy=(epy0+epy1+epy2+epy3+epy4)/5
#epy=epy0
print(epy.lev,epy.lat,'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')

epz0=epz.Fz.loc[:,0,::2,np.arange(90,0,-3)]
epz1=epz.Fz.loc[:,1,::2,np.arange(90,0,-3)]
epz2=epz.Fz.loc[:,2,::2,np.arange(90,0,-3)]
epz3=epz.Fz.loc[:,3,::2,np.arange(90,0,-3)]
epz4=epz.Fz.loc[:,4,::2,np.arange(90,0,-3)]
epz=(epz0+epz1+epz2+epz3+epz4)/5
'''


'''
ep_d=plt.contourf(lat,level,divergence,cmap=cp,extend='both')
cb = plt.colorbar(ax=ep_d, orientation="vertical", extend='both',pad=0.04, aspect=20, shrink=0.8,drawedges=True)
#horizontal横vertical竖，shrink收缩比例，ax、cax位置，aspect长宽比，pad距子图，extend两端扩充，extendfrac扩充长度，extendrect扩充形状True，spacing，
cb.set_label('m/s',size=11,rotation=0,labelpad=10)
cb.ax.tick_params(labelsize=7)
'''
# Overlay wind vectors