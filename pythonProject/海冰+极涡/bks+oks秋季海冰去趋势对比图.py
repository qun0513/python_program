import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset

#数据读取、选取
a0=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
#print('a','\n',a)
b0=a0.sic.loc[:,82.5:66.5,15.5:60.5]                      #巴伦支海海域
#print('b','\n',b)
c0=b0.mean(dim='longitude')
c0=c0.mean(dim='latitude')
year=np.arange(1979,2020)
t0=c0.sel(time=c0.time.dt.month.isin([9,10,11]))
T0=t0.sel(time=t0.time.dt.year.isin([year]))
b_Autumn=np.array(T0).reshape(41,3)                      #巴伦支海秋季 三个月份
#print(T)
#print(T[108])
#print(bs)
baut=b_Autumn.mean(axis=1)                              #巴伦支海秋季
#print('baut','\n',baut)

#去趋势，并计算标准差、以筛选高低值年份
import scipy
from scipy import signal
ba=scipy.signal.detrend(baut)                           #去趋势 ba--b autumn
  #计算标准差
vba=np.var(ba)                                          #方差  variance
sdba=np.sqrt(vba)                                       #标准差 standard deviation -sd

#数据读取、选取
a=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
#print('a','\n',a)
b=a.sic.loc[:,62.5:44.5,135.5:164.5]                    #鄂霍茨克海海域
#print('b','\n',b)
c=b.mean(dim='longitude')
c=c.mean(dim='latitude')
year=np.arange(1979,2020)
t=c.sel(time=c.time.dt.month.isin([9,10,11]))
T=t.sel(time=t.time.dt.year.isin([year]))
e_Autumn=np.array(T).reshape(41,3)                      #鄂霍茨克海秋季 三个月份
#print(T)
#print(T[108])
#print(bs)
eaut=e_Autumn.mean(axis=1)                              #鄂霍茨克海秋季
#print('baut','\n',baut)

#去趋势，并计算标准差、以筛选高低值年份
import scipy
from scipy import signal
ea=scipy.signal.detrend(eaut)                           #去趋势 ea--e autumn
  #计算标准差
vea=np.var(ea)                                          #方差  variance
sdea=np.sqrt(vea)                                       #标准差 standard deviation -sd

#求两个海域海冰的相关系数（去趋势）
import pandas as pd
bau=pd.Series(ba)                                       #au-autumn
eau=pd.Series(ea)
r=eau.corr(bau,method='pearson')
print(r)
#求两个海域海冰的相关系数
import pandas as pd
bau=pd.Series(baut)                                       #au-autumn
eau=pd.Series(eaut)
rr=eau.corr(bau,method='pearson')
print(rr)


#绘图(去趋势)
plt.rcParams['font.sans-serif']=['SimHei']                #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False                  #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)
ax.set_title('1979-2019巴伦支海和鄂霍茨克海秋季海冰去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=15)
ax.text(2008,0.11,'相关系数r=-0.0697',horizontalalignment='center',rotation=0,
        backgroundcolor='pink',fontsize=50,c='k',alpha=0.8)
#ax.set_xticks(np.arange(1979,2020,2))
#ax.axhline(y=sdmon2,lw=6,ls=':',c='r',label='sd')
#ax.axhline(y=-sdmon2,lw=6,ls=':',c='r')
plt.plot(x,ba,lw='10',c='b',label='barents')
plt.plot(x,ea,lw='10',c='r',label='okhotsk')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='best',fontsize=60)
plt.show()

#print(ba)
#print(ea)
print('baut','\n',baut)
print('eaut','\n',eaut)
'''
#绘图
plt.rcParams['font.sans-serif']=['SimHei']                #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False                  #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)
ax.set_title('1979-2019巴伦支海和鄂霍茨克海秋季海冰时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=15)
ax.text(2005,0.27,'相关系数rr=0.0726',horizontalalignment='center',rotation=0,
        backgroundcolor='pink',fontsize=50,c='k',alpha=0.8)
ax.set_ylim(0,0.30)
ax.set_yticks(np.arange(0,0.35,0.05))
#ax.axhline(y=sdmon2,lw=6,ls=':',c='r',label='sd')
#ax.axhline(y=-sdmon2,lw=6,ls=':',c='r')
plt.plot(x,baut,lw='10',c='b',label='barents')
plt.plot(x,eaut,lw='10',c='r',label='okhotsk')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='upper right',fontsize=50)
plt.show()
'''