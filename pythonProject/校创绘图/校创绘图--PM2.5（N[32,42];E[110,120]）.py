import pandas as pd
from pandas import DataFrame
import numpy as np
import xarray as xr
from datetime import datetime as dt

a1=xr.open_dataset('D:/XC/CHN_gridded_pm25_1979_2019_daily.nc')
print(a1)

a0=a1.pm25.mean(dim='time')
aa=pd.DataFrame(a0)
#print(aa)
a=a1.to_dataframe()
a2=a1.pm25.mean()
#print(a2)
b=a.dropna(how='all')
c=a.dropna(how='any')
#print(a)
#print(b)
#print(c)
d=b.pm25.mean()
#print(d)
e=c.to_xarray()
#print(e)
lat1=a1['lat']
lon1=a1['lon']

#f=((e.isel(lon=slice(32,42),lat=slice(110,120))).mean(dim='lat')).mean(dim='lon')
'''h=[]
for i in np.arange(32,43):
    for j in np.arange(110,121):
       g=e.loc[['lat=i'],['lon=j']]
       h.append(g)
print(h)
data=f.to_dataframe()
print(data)'''

f=a1.pm25.loc[:,32:42,110:120]
print(f)
DATa=f.mean(dim='lat')
DAta=DATa.mean(dim='lon')
#DATA0=DAta.mean(dim='time')
#print(DATA0)
#print(DAta)
time0=DAta.time
#print(time0)
time00=time0.to_dataframe()  ###
date=time00.time
#print(time00)
#time00.time=0
#print(time00)

'''for time in time0:
    pm25=pm25.sel(time=time)
    time00.loc[time]=np.mean(pm25)
time00['date']=time00.index
print(time00)'''

Data=DAta.to_dataframe()
print(Data)
print(Data.pm25)
#data={'Data.pm25':Data.pm25,'date':date}
frame=DataFrame(Data.pm25,columns=['pm25','time'])
print(frame)
frame.time=date
print(frame)
print('xxxxxxxxxxxxxxxxxx')
frame.time=pd.to_datetime(frame.time,format='%Y-%M-%D')

'''data1=[]
for x in np.arange(1979,2020):
    T_year = frame[frame.time.dt.year.isin([x])]
    time= T_year[T_year.time.dt.month.isin([1,2])]
    time1=T_year[T_year.time.dt.month.isin([12])]
    data2=time.pm25
    data1.append(np.array(data2))
    from numpy import *
    data1=mean(data1)
    print(data1)
    #print(sum(data1)/len(data1))
    '''
w=[]
for x in np.arange(1979,2020,1):
  T_year = a1.pm25.loc[a1.pm25.time.dt.year.isin([x])]  # or f.loc
  x=T_year.loc[T_year.time.dt.month.isin([1])]
  y=T_year.loc[T_year.time.dt.month.isin([2])]
  z=T_year.loc[T_year.time.dt.month.isin([12])]
  x=np.mean(x)
  y=np.mean(y)
  z=np.mean(z)
  w.append(np.array(x))
  w.append(np.array(y))
  w.append(np.array(z))
print(w)
