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

tt1=[]  #控制实验
for filename in os.listdir(path1):
    full_path = os.path.join(path1,filename)
    t1 = xr.open_dataset(full_path,decode_times=False)
    tt1.append(t1.ts[0,0,:,:])#.mean(axis=2))     #变量
print(t1.su)
ttt1=np.array(tt1).reshape(360,115,182)

ff1=[]  #控制实验
ff2=[]
for filename in os.listdir(path1):
    full_path = os.path.join(path1,filename)
    f1 = xr.open_dataset(full_path,decode_times=False)
    ff1.append(f1.us[0,0,:,:])#.mean(axis=2))     #变量
    ff2.append(f1.vs[0,0,:,:])

fff1=np.array(ff1).reshape(360,115,182)
fff2=np.array(ff2).reshape(360,115,182)

tt2=[]  #风应力实验
for filename in os.listdir(path2):
    full_path = os.path.join(path2,filename)
    t2 = xr.open_dataset(full_path,decode_times=False)
    tt2.append(t2.ts[0,0,:,:])#.mean(axis=2))     #变量
#print('xx',t3.su[:,50:70,160:180]) #第多少个
ttt2=np.array(tt2).reshape(360,115,182)

ff3=[]  #控制实验
ff4=[]
for filename in os.listdir(path2):
    full_path = os.path.join(path2,filename)
    f3 = xr.open_dataset(full_path,decode_times=False)
    ff3.append(f3.us[0,0,:,:])#.mean(axis=2))     #变量
    ff4.append(f3.vs[0,0,:,:])

fff3=np.array(ff3).reshape(360,115,182)
fff4=np.array(ff4).reshape(360,115,182)

#将 [-180,180]的数据，转换为[0，360]
f0=np.empty((360,115,182))
f2=np.empty((360,115,182))
t=np.empty((360,115,182))

for i in np.arange(0,182):
    if i<91:                           #111+
        f0[:,:,i]=fff1[:,:,91+i]        #+71=182
        f2[:, :, i] = fff2[:, :, 91 + i]
        t[:, :, i] = ttt1[:, :, 91 + i]

    if i>91:
        f0[:,:,i]=fff1[:,:,i-91]
        f2[:, :, i] = fff2[:, :, i - 91]
        t[:, :, i] = ttt1[:, :, i - 91]

lev=f1.ts.lev
lat=f1.ts.lat
lon=f1.ts.lon

fig=plt.figure()
proj=ccrs.PlateCarree()#central_longitude=180

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cmap1=cmaps.BlueDarkRed18#BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')


''' #全球-------------------
def HT():

    #gridlines()改变中心经度后，不太方便修改，遂放弃，转ax
    ax.set_extent([-180, 180, -90, 90],crs=proj)
    ax.coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
    ax.tick_params(size=6, labelsize=18)
    ax.set_xlabel('longitude',size=18)
    ax.set_ylabel('latitude',size=18)

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks([0,60,120,180,-120,-60,0], crs=proj)
    ax.set_yticks(np.arange(-60,61,30), crs=proj)

#子图1-------------------------------
ax=fig.add_subplot(121,projection=proj)
HT()
ax.contourf(lon,lat,t[5,:,:], cmap=cmap1, levels=np.linspace(-4, 32, 10))
ax.quiver(lon[::2],lat[::2],f0[5,::2,::2],f2[5,::2,::2],scale=20,
          headwidth=3,headlength=5,headaxislength=4.0,width=0.0015)#streamplot: ,density=1.5,linewidth=1.5
ax.annotate(str(6),xy=(-0.08,1.0),xycoords='axes fraction',c='b',fontsize=20,ha='left',va='top')
boxa = patches.Rectangle((35,-20), 60, 35, linewidth=2, linestyle='--', zorder=2,
                             edgecolor='g', facecolor='none',transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
ax.add_patch(boxa)
boxa = patches.Rectangle((180,-20), 80, 40, linewidth=2, linestyle='--', zorder=2,
                             edgecolor='b', facecolor='none',
                             transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
ax.add_patch(boxa)

#子图2---------------------------------
ax=fig.add_subplot(122,projection=proj)
HT()
ax.contourf(lon,lat,t[11,:,:], cmap=cmap1, levels=np.linspace(-4, 32, 10))


ax.quiver(lon[::2],lat[::2],f0[11,::2,::2],f2[11,::2,::2],scale=20,
          headwidth=3,headlength=5,headaxislength=4.0,width=0.0015)#streamplot: ,density=1.5,linewidth=1.5
#scale与quiver呈反比例
#headwidth 和 headlength：设置箭头头部的宽度和长度。
#headaxislength：设置箭头头部轴的长度
#headaxislength: 箭头头部轴的长度，相对于箭头长度的比例

ax.annotate(str(12),xy=(-0.08,1.0),xycoords='axes fraction',c='b',fontsize=20,ha='left',va='top')
boxa = patches.Rectangle((35,-15), 60, 35, linewidth=2, linestyle='--', zorder=2,
                             edgecolor='g', facecolor='none',
                             transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
ax.add_patch(boxa)
boxa = patches.Rectangle((180,-20), 80, 40, linewidth=2, linestyle='--', zorder=2,
                             edgecolor='b', facecolor='none',
                             transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
ax.add_patch(boxa)

#colorbar------------------------------
ax0=fig.add_axes([0.2,0.2,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-4, vmax=32)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature', size=3, rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-4, 32, 10))
fc1.ax.tick_params(labelsize=15)   #调用colorbar的ax属性

plt.suptitle('控制实验海表流速和温度分布图',fontsize=20,x=0.5,y=0.83)
'''

#太平洋---------------------------
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
        ax.set_xlabel('')
        ax.set_xticks([])
    if i==0 or i == 1  or i == 3 or i == 4:
        ax.set_ylabel('')
        ax.set_yticks([])
    if i == 0 or i==2 or i==6:
        ax.set_ylabel('')
    if i == 2 or i == 4:
        ax.set_xlabel('')
    if i==2:      #设置ylabel
        #ax.text(-0.5, 0.5, 'latitude', rotation=90, size=20, c='b')
        ax.annotate('latitude', xy=(-0.3, 1.6), xycoords='axes fraction', c='r', fontsize=20, ha='left', va='top',rotation=90)

#子图1------------------控制实验
ax=fig.add_subplot(231,projection=proj)  #加投影之后才能设置经纬度范围
i=6
HT()
ax.contourf(lon,lat,ttt1[6,:,:], cmap=cmap1,levels=np.linspace(12,32,11))
ax.quiver(lon[::],lat[::],fff1[6,::,::],fff2[6,::,::],scale=10,
          headwidth=3,headlength=5,headaxislength=4.0,width=0.0015)
#ax.set_title('控制实验海表流速和温度分布图_7月',fontsize=20,x=0.5,y=0.90)
ax.annotate(str(7),xy=(-0.1,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')

#子图2-6-----------------风应力实验
k=[0,6,66,126,354]
for i in np.arange(0,5):
    j=232+i
    ax=fig.add_subplot(j,projection=proj)  #加投影之后才能设置经纬度范围
    HT()
    ax.contourf(lon,lat,ttt2[i,:,:], cmap=cmap1,levels=np.linspace(12,32,11))
    ax.quiver(lon[::],lat[::],fff3[i,::,::],fff4[i,::,::],scale=10,
          headwidth=3,headlength=5,headaxislength=4.0,width=0.0015)
    #ax.set_title('风应力实验海表流速和温度分布图_7月',fontsize=20,x=0.5,y=0.90)
    l=k[i]+1
    ax.annotate(str(l),xy=(-0.1,1.1),xycoords='axes fraction',c='b',fontsize=20,ha='left',va='top')


#colorbar
ax0=fig.add_axes([0.2,0.08,0.6,0.03])
norm =mpl.colors.Normalize(vmin=16, vmax=32)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature', size=3, rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(12,32,11))
fc1.ax.tick_params(labelsize=15)   #调用colorbar的ax属性

plt.suptitle('热带太平洋',fontsize=20,x=0.5,y=0.95)

plt.show()


