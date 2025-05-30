import numpy as np 
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
from scipy.stats import pearsonr,mstats


a=xr.open_dataset('D:/decadal prediction/data/tos_Omon_MIROC6_dcppA-assim_r1i1p1f1_gn_195001-201712.nc',decode_times=False)
b=xr.open_dataset('D:/decadal prediction/data/tos_Omon_MIROC6_dcppA-hindcast_s1960-r1i1p1f1_gn_196011-197012.nc',decode_times=False)
c=xr.open_dataset('D:/decadal prediction/data/tos_Oday_MIROC6_dcppA-hindcast_s1960-r1i1p1f1_gn_19601101-19691231.nc',decode_times=False)
a['time']=pd.date_range(start='1/1950',end='1/2018',freq='m')  #更改时间维的命名方式  end='1/2051'  periods=432
b['time']=pd.date_range(start='11/1960',end='1/1971',freq='m')  #更改时间维的命名方式  end='1/2051'  periods=432

d=xr.open_dataset('D:/decadal prediction/data/tos_Omon_NorCPM1_dcppA-assim_r1i1p1f1_gn_195001-201812.nc',decode_times=False)
e=xr.open_dataset('D:/decadal prediction/data/tos_Omon_NorCPM1_dcppA-hindcast_s1960-r1i1p1f1_gn_196010-197012.nc',decode_times=False)

f=xr.open_dataset('D:/decadal prediction/data/tos_Omon_MRI-ESM2-0_dcppA-assim_r1i1p1f1_gn_196001-201912.nc',decode_times=False)
g=xr.open_dataset('D:/decadal prediction/data/tos_Omon_MRI-ESM2-0_dcppA-hindcast_s1960-r1i1p1f1_gn_196011-196512.nc',decode_times=False)

h=xr.open_dataset('D:/decadal prediction/data/tos_Omon_CanESM5_dcppA-assim_r1i1p2f1_gn_195801-201612.nc',decode_times=False)
i=xr.open_dataset('D:/decadal prediction/data/tos_Omon_CanESM5_dcppA-hindcast_s1960-r1i1p2f1_gn_196101-197012.nc',decode_times=False)
j=xr.open_dataset('D:/decadal prediction/data/tos_Oday_CanESM5_dcppA-hindcast_s1960-r1i1p2f1_gn_19610101-19701231.nc',decode_times=False)

k=xr.open_dataset('D:/decadal prediction/data/tas_Amon_MIROC6_dcppA-assim_r1i1p1f1_gn_195001-201712.nc',decode_times=False)
l=xr.open_dataset('D:/decadal prediction/data/tas_Amon_MIROC6_dcppA-assim_r5i1p1f1_gn_195001-201712.nc',decode_times=False)

print(a.tos)
print(b.tos)
z1=c.tos.data[0:1,:,:].mean(axis=0)
z2=j.tos.data[0:1,:,:].mean(axis=0)

t1=a.tos[130,:,:].data.ravel()
t2=b.tos[0,:,:].data.ravel()
t3=z1.ravel()
t4=d.tos[129,:,:].data.ravel()
t5=e.tos[0,:,:].data.ravel()
t6=f.tos[10,:,:].data.ravel()
t7=g.tos[0,:,:].data.ravel()
t8=h.tos[36,:,:].data.ravel()
t9=i.tos[0,:,:].data.ravel()
t10=z2.ravel()


#print(a.tos[130,:,:].time)
#rint(b.tos[0,:,:].time)

x=np.ma.masked_where(np.isnan(t1),t1)
y=np.ma.masked_where(np.isnan(t2),t2)
z=np.ma.masked_where(np.isnan(t3),t3)
u=np.ma.masked_where(np.isnan(t4),t4)
v=np.ma.masked_where(np.isnan(t5),t5)

r,p=mstats.pearsonr(x,y)
r1,p1=mstats.pearsonr(x,z)
np.savetxt('t1.txt',t1,fmt='%1.8f')
np.savetxt('t2.txt',t2,fmt='%1.8f')
np.savetxt('t3.txt',t3,fmt='%1.8f')
np.savetxt('t4.txt',t4,fmt='%1.8f')
np.savetxt('t5.txt',t5,fmt='%1.8f')
np.savetxt('t6.txt',t6,fmt='%1.8f')
np.savetxt('t7.txt',t7,fmt='%1.8f')
np.savetxt('t8.txt',t8,fmt='%1.8f')
np.savetxt('t9.txt',t9,fmt='%1.8f')
np.savetxt('t10.txt',t10,fmt='%1.8f')

np.savetxt('t11.txt',k.tas[1,:,:].data.ravel(),fmt='%1.8f')
np.savetxt('t12.txt',k.tas[1,:,:].data.ravel(),fmt='%1.8f')
#print(t1,t2)
print(r,r1)



#! ravel()函数会按照C语言的行优先顺序（C order）展平数组。如果你想要按照Fortran的列优先顺序（Fortran order）展平数组，可以指定order参数为'F'：
# flat_arr_F = arr.ravel('F')  ravel()是在数组上进行操作，同一的；flatten（）是在副本上进行操作，是分离的。

'''
# 假设我们要将小于某个数值的数据替换为nan，这里以数值0为例
threshold =-1000
# 选择需要处理的数据变量，这里以'var_name'为例，你需要替换成实际的变量名
data_var = c.tos
# 使用xarray的where函数，将小于阈值的数据替换为nan
data_var = xr.where(data_var > threshold, data_var, np.nan)
'''

a1=a.tos.mean(dim='time')
b1=b.tos.mean(dim='time')

#print(b1.shape)
#print(a.tas.data[:,45:135,::2])
#print(a.tas.data[:,45:135,::2].shape)

"""
lon=a.tos.lon
lat=a.tos.lat
#print(lon.shape)
from cartopy.util import add_cyclic_point
a1,lon=add_cyclic_point(a1,coord=lon)

proj=ccrs.PlateCarree()#central_longitude=180
fig=plt.figure()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cmap1=cmaps.BlueDarkRed18#BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

ax=fig.add_subplot(projection=proj)#central_longitude=180)           #之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lat,a1, cmap=cmap1)   #, levels=np.linspace(-4, 32, 10)
ax.set_extent([-180, 180, 0, 360],crs=proj)
ax.coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('longitude',size=18)
ax.set_ylabel('latitude',size=18)

ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([-180,120,-60,0,60,-120,180],crs=proj)   #[ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(-90,91,30), crs=proj)

plt.show()
"""
#print(l.attrs,'\n')


#print('a','\n',a.attrs)
#print('b','\n',b)
#print('c','\n',c.attrs['comment'])
#print('d','\n',d.attrs['comment'])
#print('e','\n',e)
#print('f','\n',f)
#print('g','\n',g.attrs)#.attrs['tracking_id'])
#print('h','\n',h)#.attrs['tracking_id'])

#print('i','\n',i.attrs)
#print('j','\n',j)
#print('k','\n',k.attrs['comment'])
