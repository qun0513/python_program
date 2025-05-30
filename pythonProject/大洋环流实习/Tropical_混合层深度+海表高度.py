import os
import xarray as xr
import numpy as np
import math
import  matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib import patches

path1 = 'D:/dyhl/data_60_89_CTL'         #控制实验
path2 = 'D:/dyhl/data_60_89_Wind'        #风应力实验
#path3 = 'D:/dyhl/data_60_89_Water'       #淡水通量实验

#数据读取--------------------------------------------------------------

mld1=[]  #控制实验 数据从0-90E-180E-0
z1=[]
t1=[]
for filename in os.listdir(path1):
    full_path = os.path.join(path1,filename)
    f1 = xr.open_dataset(full_path,decode_times=False)
    mld1.append(f1.mld[0,58:63,90:140].mean(dim='lat'))#.mean(dim='') .mean(axis=2))     #变量
    z1.append(f1.z0[0,58:63,90:140].mean(dim='lat'))
    t1.append(f1.ts[0,:,58:63, 90:140].mean(dim='lat'))
#print(f1.mld)
#print(f1.mld[0,58:63,90:140].mean(dim='lat'))
mld=np.array(mld1).reshape(360,50)
z0=np.array(z1).reshape(360,50)
ts=np.array(t1).reshape(360,30,50)
print(ts)

mld1=[]  #风应力实验
z1=[]
t1=[]
for filename in os.listdir(path2):
    full_path = os.path.join(path2,filename)
    f1 = xr.open_dataset(full_path,decode_times=False)
    mld1.append(f1.mld[0,58:63,90:140].mean(dim='lat'))#.mean(dim='') .mean(axis=2))     #变量
    z1.append(f1.z0[0,58:63,90:140].mean(dim='lat'))
    t1.append(f1.ts[0,:,58:63, 90:140].mean(dim='lat'))

mldw=np.array(mld1).reshape(360,50)
z0w=np.array(z1).reshape(360,50)
tsw=np.array(t1).reshape(360,30,50)
print(mldw)
'''
def HT():

    #gridlines()改变中心经度后，不太方便修改，遂放弃，转ax
    ax.set_extent([-180, -85, -25, 25],crs=proj)
    ax.coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
    ax.tick_params(size=6, labelsize=18)
    ax.set_xlabel('longitude',size=18)
    ax.set_ylabel('latitude',size=18)

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks([-180,-160,-140,-120,-100], crs=proj)
    ax.set_yticks(np.arange(-25,21,5), crs=proj)

    if i == 0 or i == 1 or i==6:
        ax.set_xlabel(' ')
        ax.set_xticks([])
    if i==0 or i == 1  or i == 3 or i == 4:
        ax.set_ylabel(' ')
        ax.set_yticks([])
    #if i == 0 or i == 6:
    #    ax.set_ylabel('')
    if i == 2 or i == 4:
        ax.set_xlabel('')
'''
lev=f1.ts.lev
lat=f1.ts.lat
lon=f1.ts[0,:,58:63, 90:140].lon

fig=plt.figure()
proj=ccrs.PlateCarree()#central_longitude=180

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cmap1=cmaps.BlueDarkRed18#BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

'''
#子图1------------------控制实验
ax=fig.add_subplot(231)  #加投影之后才能设置经纬度范围
ax.plot(np.arange(0,50),z0[6,:]*100,c='r')
ax.annotate(str(7),xy=(-0.11,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')
ax.set_xticks([])
ax.set_yticks([40,50,60,70,80])
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('')
ax.set_xticks(np.arange(0,60,10))
ax.set_xticklabels([-180,-160,-140,-120,-100,-80])

ax=fig.add_subplot(232)  #加投影之后才能设置经纬度范围
ax.plot(np.arange(0,50),z0w[0,:]*100,c='b')
ax.annotate(str(1),xy=(-0.11,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')
ax.set_xticks([])
ax.set_yticks([40,50,60,70,80])
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('')
ax.set_xticks(np.arange(0,60,10))
ax.set_xticklabels([-180,-160,-140,-120,-100,-80])

ax=fig.add_subplot(233)  #加投影之后才能设置经纬度范围
ax.plot(np.arange(0,50),z0w[6,:]*100,c='b')
ax.annotate(str(7),xy=(-0.11,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')
ax.set_xticks([])
ax.set_yticks([40,50,60,70,80])
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('')
ax.set_xticks(np.arange(0,60,10))
ax.set_xticklabels([-180,-160,-140,-120,-100,-80])

ax=fig.add_subplot(234)  #加投影之后才能设置经纬度范围
ax.plot(np.arange(0,50),z0w[66,:]*100,c='b')
ax.annotate(str(67),xy=(-0.11,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')
ax.set_xticks([])
ax.set_yticks([40,50,60,70,80])
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('')
ax.set_xticks(np.arange(0,60,10))
ax.set_xticklabels([-180,-160,-140,-120,-100,-80])

ax=fig.add_subplot(235)  #加投影之后才能设置经纬度范围
ax.plot(np.arange(0,50),z0w[126,:]*100,c='b')
ax.annotate(str(127),xy=(-0.11,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')
ax.set_xticks([])
ax.set_yticks([40,50,60,70,80])
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('')
ax.set_xticks(np.arange(0,60,10))
ax.set_xticklabels([-180,-160,-140,-120,-100,-80])

ax=fig.add_subplot(236)  #加投影之后才能设置经纬度范围
ax.plot(np.arange(0,50),z0w[354,:]*100,c='b')
ax.annotate(str(355),xy=(-0.11,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')
ax.set_xticks([])
ax.set_yticks([40,50,60,70,80])
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('')
ax.set_xticks(np.arange(0,60,10))
ax.set_xticklabels([-180,-160,-140,-120,-100,-80])

#k=[0,6,66,126,354]
'''

ax=fig.add_subplot(231)   #之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lev, ts[6, :, :], cmap=cmap1, levels=np.linspace(-4, 32, 10))
# ax.set_title(' ', pad=12)
ax.set_xlabel('Longitude(degrees)', fontsize=20, labelpad=10)
ax.set_ylabel('Depth(meters)', fontsize=20, labelpad=10)
ax.tick_params(size=6, labelsize=15)
ax.set_xticks(np.arange(-180, -80, -20))

ax.set_yscale('symlog')
# ax.set_ylim(-5010,-10)
ax.set_yticks([-5005, -2555, -1005, -505, -205, -105, -55, -25, -15])
ax.set_yticklabels([-5000, -2500, -1000, -500, -200, -100, -50, -20, -10])
ax.set_xlabel(' ')
ax.set_xticks([])

ax.annotate(str(7),xy=(-0.10,1.0),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')

cmarker = plt.contour(lon, lev, ts[6, :, :], [4, 20], colors='b')
plt.clabel(cmarker, fontsize=16, colors=['k', 'k'], fmt='%.2f')

k=[0,6,66,126,354]
for i in np.arange(0,5):
    j=232+i

    ax=fig.add_subplot(j)   #之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
    ax.contourf(lon, lev, tsw[k[i], :, :], cmap=cmap1, levels=np.linspace(-4, 32, 10))
    # ax.set_title(' ', pad=12)
    ax.set_xlabel('Longitude(degrees)', fontsize=20, labelpad=10)
    ax.set_ylabel('Depth(meters)', fontsize=20, labelpad=10)
    ax.tick_params(size=6, labelsize=15)
    ax.set_xticks(np.arange(180,278,20))
    ax.set_xticklabels(np.arange(-180, -80, 20))

    ax.set_yscale('symlog')
    # ax.set_ylim(-5010,-10)
    ax.set_yticks([-5005, -2555, -1005, -505, -205, -105, -55, -25, -15])
    ax.set_yticklabels([-5000, -2500, -1000, -500, -200, -100, -50, -20, -10])

    if i == 0 or i == 1:
        ax.set_xlabel(' ')
        ax.set_xticks([])
    if i==0 or i == 1  or i == 3 or i == 4:
        ax.set_ylabel(' ')
        ax.set_yticks([])
    #if i == 0 or i == 6:
    #    ax.set_ylabel('')
    if i == 2 or i == 4:
        ax.set_xlabel('')
    ax.annotate(str(k[i]+1), xy=(-0.12, 1.0), xycoords='axes fraction', c='b', fontsize=20, ha='left', va='top')
    cmarker = plt.contour(lon, lev, tsw[k[i], :, :], [4, 20], colors='b')
    plt.clabel(cmarker, fontsize=16, colors=['k', 'k'], fmt='%.2f')

plt.suptitle('太平洋南北纬2度内温度的垂直分布',fontsize=20,x=0.5,y=0.95)

ax0=fig.add_axes([0.2,0.08,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-4, vmax=32)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature', size=3, rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-4, 32, 10))
fc1.ax.tick_params(labelsize=15)   #调用colorbar的ax属性
plt.show()

plt.show()