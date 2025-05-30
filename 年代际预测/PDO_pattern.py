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
import cartopy.feature as cfeature

data=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
pattern=np.loadtxt("D:/decadal prediction/results/PDOpattern_HadISST1950-2018.txt")

lon=data.sst.longitude[110:260]                          
lat=data.sst.latitude[20:70]

pattern[:,70]=pattern[:,71]                                        #循环补全，相同的效果

#绘图------------------------------------------------------------------------------------------------------------------------
fig=plt.figure() #

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,20))  
cp=ListedColormap(list_cmap1,name='cp')

proj=ccrs.PlateCarree(central_longitude=180)                                            #!   central_longitude=180，只用给子图属性一次即可，不然(可能)后面会重置
ax=fig.add_subplot(111,projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lat, pattern[:,:], cmap=cp,crs=ccrs.PlateCarree())#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
ax.add_feature(cfeature.LAND, facecolor='white')
#ax.set_extent([110,150,20,70],crs=ccrs.PlateCarree() )                               #!不如 ‘数据+投影’  直接定位 --北太平洋区域
ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=24)
ax.set_xlabel('longitude',size=24,labelpad=15)
ax.set_ylabel('latitude',size=24,labelpad=15)
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
ax0=fig.add_axes([0.20,0.10,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-1, vmax=1)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature',  rotation=0, labelpad=10, fontsize=24,loc='center')
ax1=fc1.ax                                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-1, 1, 11))
fc1.ax.tick_params(labelsize=15)      #调用colorbar的ax属性

#子图位置和标题-------------------------------------
plt.subplots_adjust(left=0.120,
                    bottom=0.180,
                    right=0.920,
                    top=0.95,
                    wspace=0.1,      #子图间水平距离
                    hspace=0.15     #子图间垂直距离
                   )
plt.suptitle('PDO_1950-2018', fontsize=30, x=0.5, y=0.93)# x,y=0.5,0.98(默认)

plt.show()