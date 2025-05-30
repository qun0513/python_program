import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset

a=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
#print('a','\n',a)
b=a.sic.loc[:,82.5:66.5,15.5:60.5]    #巴伦支海海域
#print('b','\n',b)
c=b.isel(time=slice(1319,1802))       #1979-2020 (1979.12-2020.02)
#print('c','\n',c)
c=c.mean(dim='longitude')
c=c.mean(dim='latitude')
#print(c[11])

mon_12=c[0::12]                       #12月原始数据
mon_1=c[1::12]
mon_2=c[2::12]

winter0=np.empty((41,3))
winter0[:,0]=mon_12
winter0[:,1]=mon_1
winter0[:,2]=mon_2
winter1=winter0.mean(axis=1)          #冬季三月求平均
#print('winter','\n',winter1)

##################################################################################################
import scipy
from scipy import signal
mon12=scipy.signal.detrend(mon_12)    #去趋势
mon1=scipy.signal.detrend(mon_1)
mon2=scipy.signal.detrend(mon_2)
winter=scipy.signal.detrend(winter1)
#print(mon12)

#计算标准差
vwinter=np.var(winter)                #方差  variance
vmon12=np.var(mon12)
vmon1=np.var(mon1)
vmon2=np.var(mon2)
sdwinter=np.sqrt(vwinter)             #标准差 standard deviation -sd
sdmon12=np.sqrt(vmon12)
sdmon1=np.sqrt(vmon1)
sdmon2=np.sqrt(vmon2)
#print(sdmon12)

def  S(mon12,sdmon12):    #筛选 screen  高低值年份
    h0=0;l0=0;h1=0;l1=0
    for i in np.arange(0,41):
        year=0
        if mon12[i]>sdmon12:
            year=1979+i
            h0=h0+mon12[i]
            h1=h1+1
            print('high',year,mon12[i])
        if mon12[i]<-sdmon12:
            year=1979+i
            l0=l0+mon12[i]
            l1=l1+1
            print('low',year,mon12[i])
    h=h0/h1;l=l0/l1
    print(h,l)
#print('巴伦支海12月份',S(mon12,sdmon12),'\n')
#print('巴伦支海1月份',S(mon1,sdmon1),'\n')
#print('巴伦支海2月份',S(mon2,sdmon2),'\n')
#print('巴伦支海冬季',S(winter,sdwinter),'\n')

#一元线性回归
Sxy=0;Sxx=0
bw=winter1.mean()                             #   y 平均
ewinter=np.arange(1979,2020)
ew=ewinter.mean()                             #   x 平均
for i in np.arange(0,41):
    Sxy=Sxy+winter1[i]*ewinter[i]
    Sxx=Sxx+winter1[i]**2
r=(Sxy-41*bw*ew)/(Sxx-41*ew**2)
#print(r)


#绘图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)

ax.set_title('1979-2019巴伦支海冬季海冰去趋势时间序列图',fontsize=90,pad=30)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=60,size=20)
#ax.set_xticks(np.arange(1979,2020,2))
ax.axhline(y=sdmon12,lw=6,ls=':',c='r',label='sd')
ax.axhline(y=-sdmon12,lw=6,ls=':',c='r')

plt.plot(x,winter,lw='10',c='b',label='sic')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='upper left',fontsize=70)
plt.show()
#plt.savefig('D:/GC/巴伦支海12月份海冰去趋势时间序列图.png')


'''
#一元线性回归(最小二乘法)
Sxy=0;Sxx=0
bw=winter1.mean()                             #   y 平均
ewinter=np.arange(1979,2020)
ew=ewinter.mean()                             #   x 平均
for i in np.arange(0,41):
    Sxy=Sxy+winter1[i]*ewinter[i]
    Sxx=Sxx+ewinter[i]**2
k=(Sxy-41*bw*ew)/(Sxx-41*ew**2)               #-0.004855927159526892
print(k)
bb=bw-k*ew                                    #10.162186380248196
print(bb)


#绘图
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

fig,ax=plt.subplots(dpi=50,figsize=(50,45))
x=np.arange(1979,2020,1)
y=k*x+bb
ax.set_title('1979-2019巴伦支海冬季海冰时间序列图',fontsize=80,pad=20)
ax.set_xlabel('年份',fontsize=70)
ax.set_ylabel('海冰密集度',fontsize=70)
ax.tick_params(labelsize=70,size=20)
ax.text(1988,0.30,'y=-0.0049x+10.1622',horizontalalignment='center',rotation=0,
        backgroundcolor='pink',fontsize=60,c='k',alpha=0.8)
#ax.set_xticks(np.arange(1979,2020,2))
#ax.axhline(y=sdmon12,lw=6,ls=':',c='r',label='sd')
#ax.axhline(y=-sdmon12,lw=6,ls=':',c='r')

plt.plot(x,winter1,lw='10',c='b',label='sic')
plt.plot(x,y,lw='10',c='r',label='回归方程')
#plt.rcParams.update({'font.size':18})
plt.legend(loc='upper right',fontsize=60)
plt.show()
#plt.savefig('D:/GC/巴伦支海12月份海冰去趋势时间序列图.png')
'''





'''
w=[]
x1=0;y1=0;z1=0
for x in np.arange(1979,2020,1):
  T_year = c.loc[c.time.dt.year.isin([x])]  # or f.loc
  x=T_year.loc[T_year.time.dt.month.isin([1])]
  y=T_year.loc[T_year.time.dt.month.isin([2])]
  z=T_year.loc[T_year.time.dt.month.isin([12])]

  w.append(x)
  w.append(y)
  w.append(z)
  w=xr.concat(w,dim='time')

t=np.array(w).reshape(41,3)
#print('w','\n',w)
#print('t','\n',t)
'''


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
'''
