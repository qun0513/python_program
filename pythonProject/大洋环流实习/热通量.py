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
#path2 = 'D:/dyhl/data_60_89_Wind'        #风应力实验
path3 = 'D:/dyhl/data_60_89_Water'       #淡水通量实验

#数据读取--------------------------------------------------------------


lthf1=[]  #控制实验 数据从0-90E-180E-0
sshf1=[]
lwv1=[]
swv1=[]
for filename in os.listdir(path1):
    full_path = os.path.join(path1,filename)
    f1 = xr.open_dataset(full_path,decode_times=False)
    lthf1.append(f1.lthf[0, :, :])
    sshf1.append(f1.sshf[0, :, :])
    lwv1.append(f1.lwv[0, :, :])
    swv1.append(f1.swv[0, :, :])

lthf1=np.array(lthf1).reshape(360,115,182)
sshf1=np.array(sshf1).reshape(360,115,182)
lwv1=np.array(lwv1).reshape(360,115,182)
swv1=np.array(swv1).reshape(360,115,182)

hh1=lthf1+sshf1+lwv1+swv1
h1=np.empty((360,115,182))
for i in np.arange(0,182):
    if i<91:
        h1[:,:,i]=hh1[:,:,91+i]
    if i>91:
        h1[:,:,i]=hh1[:,:,i-91]

lthf2=[]  #风应力实验
sshf2=[]
lwv2=[]
swv2=[]
for filename in os.listdir(path3):
    full_path = os.path.join(path3,filename)
    f1 = xr.open_dataset(full_path,decode_times=False)
    lthf2.append(f1.lthf[0, :, :])
    sshf2.append(f1.sshf[0, :, :])
    lwv2.append(f1.lwv[0, :, :])
    swv2.append(f1.swv[0, :, :])

lthf2=np.array(lthf2).reshape(360,115,182)
sshf2=np.array(sshf2).reshape(360,115,182)
lwv2=np.array(lwv2).reshape(360,115,182)
swv2=np.array(swv2).reshape(360,115,182)

hh2=lthf2+sshf2+lwv2+swv2
h2=np.empty((360,115,182))
for i in np.arange(0,182):
    if i<91:
        h2[:,:,i]=hh2[:,:,91+i]
    if i>91:
        h2[:,:,i]=hh2[:,:,i-91]

lev=f1.ts.lev
lat=f1.ts.lat
lon=f1.ts.lon

fig=plt.figure()
proj=ccrs.PlateCarree(central_longitude=180)#central_longitude=180

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cmap1=cmaps.BlueDarkRed18#BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

def HT():

    #gridlines()改变中心经度后，不太方便修改，遂放弃，转ax
    ax.set_extent([-180, 180, -90, 90],crs=proj)
    ax.coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
    ax.tick_params(size=6, labelsize=18)
    ax.set_xlabel('longitude',size=18)
    ax.set_ylabel('latitude',size=18)

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks([0, 60, 120, 180, -120, -60, 0], crs=proj)
    ax.set_yticks(np.arange(-60, 61, 30), crs=proj)

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

#子图1------------------控制实验
ax=fig.add_subplot(231,projection=proj)  #加投影之后才能设置经纬度范围
i=6
HT()
ax.contourf(lon,lat,h1[6,:,:], cmap=cmap1,levels=np.linspace(0,320,11))
ax.annotate(str(7),xy=(-0.1,1.1),xycoords='axes fraction',c='r',fontsize=20,ha='left',va='top')

#子图2-6-----------------风应力实验
k=[0,6,66,126,354]
for i in np.arange(0,5):
    j=232+i
    ax=fig.add_subplot(j,projection=proj)  #加投影之后才能设置经纬度范围
    HT()
    ax.contourf(lon,lat,h2[i,:,:], cmap=cmap1,levels=np.linspace(0,320,11))

    l=k[i]+1
    ax.annotate(str(l),xy=(-0.1,1.1),xycoords='axes fraction',c='b',fontsize=20,ha='left',va='top')

#colorbar
ax0=fig.add_axes([0.2,0.08,0.6,0.03])
norm =mpl.colors.Normalize(vmin=0, vmax=320)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature', size=3, rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(0,320,11))
fc1.ax.tick_params(labelsize=15)   #调用colorbar的ax属性

plt.suptitle('淡水通量实验全球海表面净热通量',fontsize=20,x=0.5,y=0.95)

plt.show()