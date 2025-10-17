 # ?三维数组的写入
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
import cartopy.feature as cfeature
import cmaps
import matplotlib as mpl
from eofs.standard import Eof

#忽略警告信息
import warnings
warnings.filterwarnings('ignore')

#数据处理---------------------------------------------------------------------------------------------------------------------
data=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
data1=pd.read_table('D:/decadal prediction/data/ersst.v5.pdo.dat.txt',sep=r'\s+',header=1)
pdo0=np.array(data1.values)
#print(data.sst[1080:1788,:,:])
pdo1=pdo0[96:165,1:13]            #1950-2018
pdo2=pdo1.reshape(828)
#print(pdo1)
sst0=np.empty((828,180,360))      #（-180,180） 转为（0，360）------------------
for i in np.arange(0,360):
    if i<180:
        sst0[:,:,i]=data.sst[960:1788,:,180+i]
    if i>=180:
        sst0[:,:,i]=data.sst[960:1788,:,i-180]

sst1=sst0
sst1[sst1==-1000]=np.nan
sst1[sst1==-1.8]=np.nan

# remove the global-mean SST anomaly----------------------
sst_g=sst1.reshape((69,12,180,360))
sst_gm=np.nanmean(sst_g,axis=0)
sst_gma=np.empty((69,12,180,360))
for j in np.arange(0,12):
    for k in np.arange(0,180):
        for l in np.arange(0,360):
            sst_gma[:,j,k,l]=sst_g[:,j,k,l]-sst_gm[j,k,l]
sst_gma=sst_gma.reshape((828,180,360))

#remove the climatological annual cycle------------------------
sst_np=sst1[:,20:70,110:260]                          #北太平洋海温   (1791,50,150)
sst_nps=sst_np.reshape((69,12,50,150))         #年循环
sst_ac=np.nanmean(sst_nps,axis=0)             #年循环气候态

sst2=np.empty((828,50,150))
sst_trend=np.empty(828)
for i in np.arange(828):
    for j in np.arange(0,50):
        for k in np.arange(0,150):
            sst2[i,j,k]=sst_np[i,j,k] -np.nanmean( sst_gma[i,:,:]  )                   #(1788,50,150)
            sst_trend[i]=np.nanmean( sst_gma[i,:,:])
np.savetxt("D:/decadal prediction/results/sst_trend_HadISST.txt",sst_trend,fmt='%1.8f')

sst3=np.empty((69,12,50,150))
for j in np.arange(0,12):
    for k in np.arange(0,50):
        for l in np.arange(0,150):
            sst3[:,j,k,l]=sst2.reshape((69,12,50,150))[:,j,k,l]-sst_ac[j,k,l]
sst=sst3.reshape((828,50,150))
with open('D:/decadal prediction/results/sst_ac_HadISST.txt', 'w') as f:
    # 遍历数组的每个切片
    for two_d_slice in sst_ac:
        # 将二维切片转换为字符串，元素之间用空格分隔
        for row in two_d_slice:
            f.write(' '.join(map(str, row)) + '\n')
        # 每个切片之间添加一个空行
        f.write('\n')
#np.savetxt("D:/decadal prediction/results/sst_ac.txt",sst_ac,fmt='%1.8f')

lat0=data.sst.latitude.loc[70:20].data
coslat=np.cos(np.deg2rad(lat0))
weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重

solver = Eof(sst, weights=weight)                        # 创建EOF求解器
EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
VAR=solver.varianceFraction(neigs=3)

#np.savetxt("D:/decadal prediction/results/PDOindex_HadISST1950-2018.txt",PC[:,0],fmt='%1.8f')
#np.savetxt("D:/decadal prediction/results/PDOpattern_HadISST1950-2018.txt",EOF[0,:,:],fmt='%1.8f')
#print(PC[:,0])
print(VAR)

pdo=pd.Series(pdo2)
pc1=pd.Series(PC[:,0])
r=pdo.corr(pc1,method='pearson')                    #相关性
print('r','\n',r)

lon=data.sst.longitude[110:260]                          #!  也很关键，要理解数据定位的区域 和 画布定位的区域，如何让二者重合
lat=data.sst.latitude[20:70]
"""
#lon=np.arange(110,260)
#lat=np.arange(20,70)
#lon,lat=np.meshgrid(lon,lat)    ---
from cartopy.util import add_cyclic_point
sst,lon=add_cyclic_point(sst, coord=lon)  #循环补全数据；最后一个由下一循环的第一个补上
sst[:,180] = sst[:, 181]    #
"""
EOF[0,:,70]=EOF[0,:,71]                                        #循环补全，相同的效果

#绘图------------------------------------------------------------------------------------------------------------------------
fig=plt.figure() #

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,20))  
cp=ListedColormap(list_cmap1,name='cp')

proj=ccrs.PlateCarree(central_longitude=180)                                            #!   central_longitude=180，只用给子图属性一次即可，不然(可能)后面会重置
ax=fig.add_subplot(111,projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lat, EOF[0,:,:], cmap=cp,crs=ccrs.PlateCarree())#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
ax.add_feature(cfeature.LAND, facecolor='white')
#ax.set_extent([110,150,20,70],crs=ccrs.PlateCarree() )                               #!不如 ‘数据+投影’  直接定位 --北太平洋区域
ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('longitude',size=18)
ax.set_ylabel('latitude',size=18)
#? 画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
#? 但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.

ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([180,-150,120,-120,150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())

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
                    bottom=0.150,
                    right=0.950,
                    top=0.92,
                    wspace=0.1,      #子图间水平距离
                    hspace=0.15     #子图间垂直距离
                   )
plt.suptitle('PDO_1950-2018', fontsize=20, x=0.5, y=0.92)# x,y=0.5,0.98(默认)

plt.show()


# better comments--------------------------------------------------------------------------------
#//! 红色的高亮注释
#//? 蓝色的高亮注释
#//* 绿色的高亮注释
#//todo 橙色的高亮注释

#// 灰色带删除线的注释

# 熟悉GitHub的修改文件等操作
