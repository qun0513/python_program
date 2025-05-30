import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import cmaps
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from eofs.standard import Eof
from PyEMD import EEMD
from scipy.signal import hilbert, periodogram
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import pearsonr

ds=xr.open_dataset("D:\pycharm\sstv3b.mnmean_18540101-20200201.nc")  #xr.open_dataset('D:/pycharm/sst.mnmean_18540101-20250301.nc')
print(ds)
koe = ds.sst.sel(
    time=slice("1950-01-01", "2011-12-01"),  #ds["time"] = xr.decode_cf(ds).time  # 自动解析时间
    lat=slice(50, 30),
    lon=slice(145, 210),
    #zlev=0
)
print('-------------',koe)
lon0=koe.lon
lat0=koe.lat
sst=koe.data
length=len(koe.time.data)
yr=length//12

sst9=sst.reshape(yr,12,len(lat0.data),len(lon0.data))
sst10=sst9.mean(axis=0)
sst11=sst9-sst10

# 全球变暖趋势
glb=ds.sst.sel(time=slice("1950-01-01", "2011-12-01"))#,  zlev=0
sst4=glb.data.reshape(yr,12,len(glb.lat.data),len(glb.lon.data))
sst5=sst4.mean(axis=0)
sst6=sst4-sst5
sst7=np.nanmean(sst6.reshape(length,len(glb.lat.data),len(glb.lon.data)),axis=(1,2))
#去趋势
sst12=sst11.reshape(length,len(lat0.data),len(lon0.data))-sst7[:,None,None]

koe_idx=np.nanmean(sst11.reshape(length,len(lat0.data),len(lon0.data)),axis=(1,2))     #?------------
koe_idx=(koe_idx-np.mean(koe_idx))/np.std(koe_idx)
#np.savetxt('D:/decadal prediction/results/piControl/HadISST/koe_index_1924-2011_ersstv3b.txt',koe_idx,fmt='%.3f')

npacific=ds.sst.sel(time=slice("1950-01-01", "2011-12-01"),  #ds["time"] = xr.decode_cf(ds).time  # 自动解析时间
    lat=slice(70, 20),
    lon=slice(120, 260),
    #zlev=0
    )
print('--------npacific---------','\n',npacific)
#print(npacific[0,0,0],ds.sst.sel(time=slice("1924-01-01", "1924-01-01"),lat=slice(20,21),lon=slice(120,121)))
#print(lat)
sst0=npacific.data
sst1=sst0.reshape(yr,12,len(npacific.lat.data),len(npacific.lon.data))
sst2=np.nanmean(sst1,axis=0)
sst3=sst1-sst2

sst8=sst3.reshape(length,len(npacific.lat.data),len(npacific.lon.data))-sst7[:,None,None]

lat=npacific.lat.data
print(lat)
coslat=np.cos(np.deg2rad(lat))
weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重
solver = Eof(sst8, weights=weight)                        
EOF= solver.eofsAsCorrelation(neofs=3)             
PC = solver.pcs(npcs=3, pcscaling=1)            
VAR=solver.varianceFraction(neigs=3)

lat=npacific.lat
lon=npacific.lon
pdo_idx=PC[:,0]
#print(len(koe_idx),len(pdo_idx))
r,p=pearsonr(koe_idx,pdo_idx)
print(r,p)




#绘图-----------------------------------------------
fig=plt.figure(figsize=(8,10),dpi=100) 

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,20))  
cp=ListedColormap(list_cmap1,name='cp')

proj=ccrs.PlateCarree(central_longitude=180)                                            
ax=fig.add_subplot(211,projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lat, EOF[0,:,:], cmap=cp,transform=ccrs.PlateCarree())#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
ax.add_feature(cfeature.LAND, facecolor='white')
ax.set_extent([120, 260, 20, 70], crs=ccrs.PlateCarree())                                 
ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=18)
#ax.set_xlabel('longitude',size=18)
#ax.set_ylabel('latitude',size=18)
#? 画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
#? 但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.

ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([180,-150,120,-120,150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())

#时间序列-------------------------------------------

l=len(koe_idx)
ll=len(koe_idx)-60
lll=len(koe_idx)-120
# 5年滑动平均，10年周期
y1=np.empty((ll))
y2=np.empty((ll))
for i in np.arange(30, len(koe_idx)-30):
    y1[i-30]=np.mean(koe_idx[i-30: i+30])  
    y2[i-30]=np.mean(pdo_idx[i-30: i+30])  
r1,p1=pearsonr(y1,y2)  # r: -0.91442178850223, -0.9062293190474227
print('r1,p1',r1,p1)

# 10年滑动平均，20年周期
y3=np.empty((lll))
y4=np.empty((lll))
for i in np.arange(60, len(koe_idx)-60):
    y3[i-60]=np.mean(koe_idx[i-60: i+60])  
    y4[i-60]=np.mean(pdo_idx[i-60: i+60])  
r2,p2=pearsonr(y3,y4)  # r: -0.8757869124963087,-0.9281345806283732
print('r2,p2',r2,p2)


ax1=fig.add_subplot(212)
ax1.plot(np.arange(0,length),PC[:,0],c='b',lw=0.6)
ax1.plot(np.arange(0,ll)+30,y2,c='b',lw=1.5)
ax1.plot(np.arange(0,length),koe_idx,c='r',lw=0.6)
ax1.plot(np.arange(0,ll)+30,y1,c='r',lw=1.5)

#ax1.plot(np.arange(0,1788),df,c='k',lw=3)
ax1.set_xticks(np.arange(0,length,240))
#ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax1.set_xlabel('year',size=18)
ax1.set_xticklabels(np.arange(1950,(1950+length//12),20))
ax1.tick_params(size=6, labelsize=18)

plt.show()




'''
a = 6357000  
pi = np.pi
#lat = np.array(ds1.sst[0,40:60,0].latitude.values)     #?-----
#print(lat)
dx = 2*pi*a*np.cos(np.deg2rad(lat0.data))/360 *2 
dy = 2*pi*a/360*2  
s_grid = (dx[:,np.newaxis]*dy)*np.ones((len(lon0.data),))    #?-----
valid_mask = ~np.isnan(sst11.reshape(length,len(lat0.data),len(lon0.data))[0])  
total_area = np.sum(s_grid*valid_mask) 
weight = s_grid/total_area 

# 月指数
koe_idx=np.nansum(sst11.reshape(length,len(lat0.data),len(lon0.data))*weight[None,:,:], axis=(1,2))
'''