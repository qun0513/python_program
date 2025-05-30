import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset

#数据读取、选取
a=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
#print('a','\n',a)
b=a.sic.loc[:,82.5:66.5,15.5:60.5]                      #巴伦支海海域
#print('b','\n',b)
c=b.mean(dim='longitude')
c=c.mean(dim='latitude')
year=np.arange(1979,2020)
t=c.sel(time=c.time.dt.month.isin([9,10,11]))
T=t.sel(time=t.time.dt.year.isin([year]))
b_Autumn=np.array(T).reshape(41,3)                      #巴伦支海秋季 三个月份
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

#筛选 screen  高低值年份
def  S(ba,sdba):                                        #筛选 screen  高低值年份
    h0=0;l0=0;h1=0;l1=0
    for i in np.arange(0,41):
        year=0
        if ba[i]>sdba:
            year=1979+i
            h0=h0+ba[i]
            h1=h1+1
            print('high',year,ba[i])
        if ba[i]<-sdba:
            year=1979+i
            l0=l0+ba[i]
            l1=l1+1
            print('low',year,ba[i])
    h=h0/h1;l=l0/l1                                     #高低值年均值
    #print(h,l)
print('巴伦支海秋季',S(ba,sdba),'\n')


#绘图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)

ax.set_title('1979-2019巴伦支海秋季海冰去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=20)
#ax.set_xticks(np.arange(1979,2020,2))
ax.axhline(y=sdba,lw=6,ls=':',c='r',label='sd')
ax.axhline(y=-sdba,lw=6,ls=':',c='r')

plt.plot(x,ba,lw='10',c='b',label='sic')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='best',fontsize=70)
plt.show()


'''
#保留趋势，并计算标准差、以筛选高低值年份
import scipy
from scipy import signal
##ba=scipy.signal.detrend(baut)                           #去趋势 ba--b autumn
  #计算标准差
vba=np.var(baut)                                          #方差  variance
sdba=np.sqrt(vba)                                         #标准差 standard deviation -sd

#筛选 screen  高低值年份
def  S(baut,sdba):                                        #筛选 screen  高低值年份
    h0=0;l0=0;h1=0;l1=0
    for i in np.arange(0,41):
        year=0
        if baut[i]>sdba:
            year=1979+i
            h0=h0+baut[i]
            h1=h1+1
            print('high',year,baut[i])
        if baut[i]<-sdba:
            year=1979+i
            l0=l0+baut[i]
            l1=l1+1
            print('low',year,baut[i])
#    h=h0/h1;l=l0/l1                                     #高低值年均值
    #print(h,l)
print('巴伦支海秋季',S(baut,sdba),'\n')

#绘图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)

ax.set_title('1979-2019巴伦支海秋季海冰去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=20)
#ax.set_xticks(np.arange(1979,2020,2))
ax.axhline(y=sdba,lw=6,ls=':',c='r',label='sd')
ax.axhline(y=-sdba,lw=6,ls=':',c='r')

plt.plot(x,baut,lw='10',c='b',label='sic')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='best',fontsize=70)
plt.show()
'''