import numpy as np
import xarray as xr
from scipy.stats import zscore
from eofs.standard import Eof
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
import matplotlib as mpl
from eofs.standard import Eof
#忽略警告信息
import warnings
warnings.filterwarnings('ignore')

import xesmf as xe
data=xr.open_dataset("D:/decadal prediction/data/hindcast/MIROC6/thetao_Omon_MIROC6_dcppA-hindcast_s1960-r1i1p1f1_gn_197001-197012.nc")
print(data)
#print(data.attrs['comment'])
target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
regridder = xe.Regridder(data, target_grid, 'bilinear', filename=None)
regridderdata= regridder(data['thetao'])
print(regridderdata.shape)
print(regridderdata)

"""
rdata=np.empty(regridderdata.shape)                                               #（-180,180） 转为（0，360）------------------
for i in np.arange(0,360):
    if i<180:
        rdata[:,:,i]=regridderdata[:,:,180+i]
    if i>180:
        rdata[:,:,i]=regridderdata[:,:,i-180]
lon=regridderdata.lon[110:160,110:260]                          #  也很关键，要理解数据定位的区域 和 画布定位的区域，如何让二者重合
lat=regridderdata.lat[110:160,110:260]                            #    经纬度都挑选那个区域
#print(lon,lat)
#rdata[:,:,180]=rdata[:,:,181]                                               #循环补全
# remove the global-mean SST anomaly----------------------

sst1=rdata[2:122,:,:]

sst_g=sst1.reshape((10,12,180,360))
sst_gm=np.nanmean(sst_g,axis=0)
sst_gma=np.empty((10,12,180,360))
for j in np.arange(0,12):
    for k in np.arange(0,180):
        for l in np.arange(0,360):
            sst_gma[:,j,k,l]=sst_g[:,j,k,l]-sst_gm[j,k,l]
sst_gma=sst_gma.reshape((120,180,360))

#remove the climatological annual cycle------------------------
sst_np=sst1[:,110:160,110:260]                          #北太平洋海温   (1791,50,150)
sst_nps=sst_np.reshape((10,12,50,150))        #年循环
sst_ac=np.nanmean(sst_nps,axis=0)             #年循环气候态

sst2=np.empty((120,50,150))
for i in np.arange(120):
    for j in np.arange(0,50):
        for k in np.arange(0,150):
            sst2[i,j,k]=sst_np[i,j,k] -np.nanmean( sst_gma[i,:,:]  )                   #(1788,50,150)

sst3=np.empty((10,12,50,150))
for j in np.arange(0,12):
    for k in np.arange(0,50):
        for l in np.arange(0,150):
            sst3[:,j,k,l]=sst2.reshape((10,12,50,150))[:,j,k,l]-sst_ac[j,k,l]
sst=sst3.reshape((120,50,150))

#print(regridderdata.lat.loc[160:110,0])
lat0=lat.data[:,0]
#print(lat0.shape)
coslat=np.cos(np.deg2rad(lat0))
weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重

solver = Eof(sst, weights=weight)                        # 创建EOF求解器
EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
VAR=solver.varianceFraction(neigs=3)

#np.savetxt(f"pdo_{year}r{r}.txt",PC[:,0],fmt='%1.8f')

#pdo=pd.Series(pdo2)
#pc1=pd.Series(PC[:,0])
#cc=pdo.corr(pc1,method='pearson')                    #相关性
#print(f'{year}r{r}cc',cc,'\n')
'''
将您的目标三维数据投影到第一模态上，实际上是计算目标数据与第一模态之间的点积。这会给出每个时间步的主成分得分。
projection = np.dot(target_data, first_eof)
'''

#绘图-------------------------------------------------------------------------------------
fig=plt.figure()  #每年一张图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']         # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False                     #用来正常显示负号
cmap1=cmaps.BlueDarkRed18                                        #BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,20))
cp=ListedColormap(list_cmap1,name='cp')

#subp=(3,4,r+1)
#print(subp)
proj=ccrs.PlateCarree(central_longitude=180)                   #!   central_longitude=180，只用给子图属性一次即可，不然(可能)后面会重置
ax=fig.add_subplot(3,4,1,projection=proj)                           # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lat, EOF[0,:,:] , cmap=cp)#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
#ax.set_extent([-180,180,-90,90],crs=ccrs.PlateCarree() )    #不如 ‘数据+投影’  直接定位 -->北太平洋区域--这个范围不好设置
ax.coastlines(resolution="50m", linewidth=1)                     # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('longitude',size=18)
ax.set_ylabel('latitude',size=18)
'''
画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.
'''
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([180,150,120,-120,-150,-180],crs=ccrs.PlateCarree())      #似乎很关键√√ ticks始终显示的是lon,lat的值（上两行可修正），但就看显示的是谁的点位                              # [ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())                  # 也能定画图范围,crs可略去

'''
#时间序列-------------------------------------------
ax1=fig.add_subplot(212)
ax1.plot(np.arange(0,1788),PC[:,0],c='b',lw=1)
#ax1.plot(np.arange(0,1788),df,c='k',lw=3)
ax1.set_xticks([0,359,719,1079,1439,1767])
#ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax1.set_xlabel('year')
ax1.set_xticklabels([1870,1900,1930,1960,1990,2018])
ax1.tick_params(size=6, labelsize=18)
'''
#绘制colorbar----------------------------------------
ax0=fig.add_axes([0.2,0.08,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-1, vmax=1)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
            cmap=cp),cax=ax0,
            orientation='horizontal',extend='both')
fc1.set_label('Temperature',  rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-1, 1, 11))
fc1.ax.tick_params(labelsize=15)      #调用colorbar的ax属性

#子图位置和标题-------------------------------------
plt.subplots_adjust(left=0.080,
                bottom=0.200,
                right=0.920,
                top=0.92,
                wspace=0.1,                #子图间垂直距离
                hspace=0.15                #子图间水平距离
            )
plt.suptitle('PDO', fontsize=20, x=0.5, y=0.96)# x,y=0.5,0.98(默认)
plt.show()
"""