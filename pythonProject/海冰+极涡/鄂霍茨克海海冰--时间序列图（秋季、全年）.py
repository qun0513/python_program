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

#筛选 screen  高低值年份
def  S(ea,sdea):                                        #筛选 screen  高低值年份
    h0=0;l0=0;h1=0;l1=0
    for i in np.arange(0,41):
        year=0
        if ea[i]>sdea:
            year=1979+i
            h0=h0+ea[i]
            h1=h1+1
            print('high',year,ea[i])
        if ea[i]<-sdea:
            year=1979+i
            l0=l0+ea[i]
            l1=l1+1
            print('low',year,ea[i])
    h=h0/h1;l=l0/l1                                     #高低值年均值
    #print(h,l)
print('鄂霍茨克海秋季',S(ea,sdea),'\n')

#绘图
plt.rcParams['font.sans-serif']=['SimHei']              #用来正常显示中文标签 SimHei FangSong(更普适)
plt.rcParams['axes.unicode_minus']=False                #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)

ax.set_title('1979-2019鄂霍茨克海秋季海冰去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=20)
#ax.set_xticks(np.arange(1979,2020,2))
ax.axhline(y=sdea,lw=6,ls=':',c='r',label='sd')
ax.axhline(y=-sdea,lw=6,ls=':',c='r')

plt.plot(x,ea,lw='10',c='b',label='sic')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='lower right',fontsize=70)
plt.show()