import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset

a0=xr.open_dataset('D:/GC/HadISST_sst_187001-201903.nc')  #bs-barents
print('a0','\n',a0)
bs=a0.sst.loc[:,82.5:66.5,15.5:60.5]      #巴伦支海海域
#print(bs.max())
#print('bs','\n',bs)
bst=bs.isel(time=slice(1319,1802))       #t - 1979-2020 (1979.12-2020.02)
bst111=bst[::-1]
print(bst111)
bst=bst.where(bst.data>-1.79)            #-1000是永久冰架的填充,且-1.8为冷凝点，因此取>-1.8（-1.79）
#print('bst','\n',bst)
#print(bst.max())
b=bst.mean(dim='longitude')              #barents
b=b.mean(dim='latitude')
#print('b','\n',b)
#print(c[11])


bmon_12=b[0::12]                       #12月原始数据
bmon_1=b[1::12]
bmon_2=b[2::12]

bwinter0=np.empty((40,3))
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
#print(bwinter)

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


a1=xr.open_dataset('D:/GC/HadISST_sst_187001-201903.nc')     #e-Okhotsk
#print('a','\n',a)
es=a1.sst.loc[:,62.5:44.5,135.5:164.5]  #鄂霍茨克海海域
#print('b','\n',b)
est=es.isel(time=slice(1319,1802))       #1979-2020 (1979.12-2020.02)
est=est.where(est.data>-1.79)            #-1000是永久冰架的填充,且-1.8为冷凝点，因此取>-1.8（-1.79）
#print(est.max())
#print(c)
#print('c','\n',c)
e=est.mean(dim='longitude')
e=e.mean(dim='latitude')
#print(c[11])

emon_12=e[0::12]                       #12月原始数据
emon_1=e[1::12]
emon_2=e[2::12]

ewinter0=np.empty((40,3))
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

#求两个海域海冰的相关系数
import pandas as pd
bwin=pd.Series(bwinter)
ewin=pd.Series(ewinter)
r=ewin.corr(bwin,method='pearson')
print(r)
'''
Sxy=0;Sxx=0
bw=bwinter.mean()                             #   x 平均
ew=ewinter.mean()                             #   y 平均
for i in np.arange(0,40):
    Sxy=Sxy+bwinter[i]*ewinter[i]
    Sxx=Sxx+bwinter[i]**2
r=(Sxy-41*bw*ew)/(Sxx-41*bw**2)
print(r)
'''

#绘图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2019,1)
ax.set_title('1979-2018巴伦支海和鄂霍茨克海冬季海温去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海温',fontsize=70)
ax.tick_params(labelsize=70,size=15)
ax.text(2003,0.56,'相关系数r=-0.0586',horizontalalignment='center',rotation=0,
        backgroundcolor='pink',fontsize=50,c='k',alpha=0.8)
#ax.set_xticks(np.arange(1979,2020,2))
#ax.axhline(y=sdmon2,lw=6,ls=':',c='r',label='sd')
#ax.axhline(y=-sdmon2,lw=6,ls=':',c='r')
plt.plot(x,bwinter,lw='10',c='b',label='barents')
plt.plot(x,ewinter,lw='10',c='r',label='okhotsk')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='best',fontsize=60)
plt.show()



#调试部分
T=b.sel(time=b.time.dt.month.isin([12,1,2]))
tt=T.sel(time=T.time.dt.year.isin([1996,1997,1998]))
#print(bwinter.min())
