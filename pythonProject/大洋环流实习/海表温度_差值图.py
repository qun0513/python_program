import os
import xarray as xr
import numpy as np
import math
import  matplotlib.pyplot as plt
path1 = 'D:/dyhl/data_60_89_CTL'         #控制实验
path2 = 'D:/dyhl/data_60_89_Wind'        #风应力实验
path3 = 'D:/dyhl/data_60_89_Water'       #淡水通量实验

#数据读取--------------------------------------------------------------

ff1=[]  #控制实验
for filename in os.listdir(path1):
    full_path = os.path.join(path1,filename)
    f1 = xr.open_dataset(full_path,decode_times=False)
    ff1.append(f1.ts[0,0,:,:])#.mean(axis=2))     #变量
print(f1.su)
fff1=np.array(ff1).reshape(360,115,182)

lev=f1.ts.lev
lat=f1.ts.lat
lon=f1.ts.lon

ff2=[]  #风应力实验
for filename in os.listdir(path2):
    full_path = os.path.join(path2,filename)
    f2 = xr.open_dataset(full_path,decode_times=False)
    ff2.append(f2.ts[0,0,:,:])#.mean(axis=2))     #变量
print('xx',f2.su[:,50:70,160:180]) #第多少个
fff2=np.array(ff2).reshape(360,115,182)

ff3=[]  #淡水通量实验
for filename in os.listdir(path3):
    full_path = os.path.join(path3,filename)
    f3 = xr.open_dataset(full_path,decode_times=False)
    ff3.append(f3.ts[0,0,:,:])#.mean(axis=2))     #变量
fff3=np.array(ff3).reshape(360,115,182)

#绘图变量------------------------
fff=fff3[:,:,:]-fff1[:,:,:]
#fff=fff1-18.78094816

#绘图------------------------------------------------------------------
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER,mticker
import cmaps
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from cartopy import feature as cfeature
from matplotlib import patches
import cartopy

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


cmap1=cmaps.BlueDarkRed18#BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')




proj=ccrs.PlateCarree(central_longitude=180) #投影经度 （-180，180） 级别高于set_extent()，中心经度似乎是决定数据的起始
fig=plt.figure(figsize=(50,45),dpi=100)
#绘图函数-------------------
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

    #ax.grid(linestyle='--', linewidth=0.5, color='gray')

    #ax.set_yticks(np.linspace(38, 46, 5), crs=proj)
    #ax.set_xticks([-120,-60,0,60,120])
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5:
        ax.set_xlabel(' ')
        ax.set_xticks([])
    if i == 1 or i == 2 or i == 4 or i == 5 or i == 7 or i == 8:
        ax.set_ylabel(' ')
        ax.set_yticks([])
    if i == 0 or i == 6:
        ax.set_ylabel('')
    if i == 6 or i == 8:
        ax.set_xlabel('')

    #尝试只在右边、底边加刻度（有待改善）
    '''
    if i == 0 or i == 3 or i == 6 or i ==7 or i == 8:
        gl.xlabels_top = False  # 标签
        gl.ylabels_right = False
        gl.xlabels_bottom = True
        gl.ylabels_left = True
        gl.xlocator = mticker.FixedLocator([-120,-60,0,60,120]) #由于不显示180，干脆也去掉-180
        gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    '''

#将 [-180,180]的数据，转换为[0，360]
f0=np.empty((360,115,182))
for i in np.arange(0,182):
    if i<91:
        f0[:,:,i]=fff[:,:,91+i]
    if i>91:
        f0[:,:,i]=fff[:,:,i-91]
#f0_no_nan = np.nan_to_num(f0, nan=0.0) # 将f0数组中的NaN值替换为0

#距平图的绘制
ax=fig.add_subplot(111,projection=proj)
HT()
ax.contourf(lon, lat, f0[11, :, :], cmap=cmap1, levels=np.linspace(-1.5, 1.5, 11))
#ax.set_title('淡水通量实验与控制实验的海表温度差值图_第12月',fontsize=20,x=0.5,y=1)
ax.annotate(str(12),xy=(-0.05,1.0),xycoords='axes fraction',c='b',fontsize=20,ha='left',va='top')
boxa = patches.Rectangle((-60, 50), 75, 20, linewidth=2, linestyle='--', zorder=2,
                             edgecolor='r', facecolor='none',
                             transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
ax.add_patch(boxa)
'''

#title=['T1','T2','T3','T4']
k=[5,11,35,59,99,159,239,299,349] #1,3,6,9,12,18,24,36,360
#k=np.arange(220,229)
for i in np.arange(0,9):
    j=331+i

    ax=fig.add_subplot(j,projection=proj)   #（projection）之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用

    HT()
    #l=k[i]+i
    ax.contourf(lon, lat, f0[k[i], :, :], cmap=cmap1, levels=np.linspace(-5, 5, 11))
    #ax.add_feature(cfeature.LAND,edgecolor='k', facecolor='white')  #正常来说，不需要这一步
    ax.set_title(' ', pad=12)
    #n=i+1
    m=k[i]+1
    ax.annotate(str(m),xy=(-0.10,1.0),xycoords='axes fraction',c='b',fontsize=16,ha='left',va='top')
    #淡水通量之添加
    #boxa = patches.Rectangle((-60, 50), 75, 20, linewidth=2, linestyle='--', zorder=2,
    #                         edgecolor='r', facecolor='none',
    #                         transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
    #ax.add_patch(boxa)
'''
#限定子图的位置等参数
plt.subplots_adjust(left=0.080,
                    bottom=0.165,
                    right=0.920,
                    top=0.92,
                    wspace=0.1, #子图间垂直距离
                    hspace=0.15  #子图间水平距离
                   )

plt.suptitle('淡水通量实验与控制实验的海表温度差值图_第12月',fontsize=20,x=0.5,y=0.98)# x,y=0.5,0.98(默认)

ax0=fig.add_axes([0.2,0.08,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-1.5, vmax=1.5)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature', size=3, rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-1.5, 1.5, 11))
fc1.ax.tick_params(labelsize=15)   #调用colorbar的ax属性
plt.show()

'''
def HT(variable,title):
    ax.coastlines(resolution='110m', linewidth=1)
    ax.set_extent([-30, 150, 0, 90])
    title='海温'
    ax.set_title(title, fontsize=15, pad=20)
    ax.contourf(lon, lat, variable, cmap=cp)
proj=ccrs.PlateCarree()
fig=plt.figure(figsize=(50,45),dpi=100)
ax0=fig.add_subplot(221,projection=ccrs.PlateCarree())
ax0.coastlines(resolution='110m',linewidth=1)
ax0.set_extent([-30,150,0,90])
ax0.set_title('0',fontsize=15,pad=20)
ax0.contourf(lon,lat,u,cmap=cp)

ax1=fig.add_subplot(212,projection=ccrs.PlateCarree())
ax1.coastlines(resolution="50m",linewidth=1)
ax1.set_extent([-30,150,0,90])
ax1.set_title('1',fontsize=15,pad=20)
ax1.contourf(lon,lat,v,cmap=cp)
#创建子图来放置colorbar
ax=fig.add_axes([0.2,0.2,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-6, vmax=6)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax,
                 orientation='horizontal',extend='both')

plt.suptitle('0123')
plt.show()
'''
#-------------------------------
'''
#subfig=np.arange(331,340)
#nmonth=np.arange(0,360,36)
for i in subfig:
    j=i-331
    plt.subplots(i,)
    plt.contourf(lon,lat,fff[nmonth[j],:,:],cmap=cp,levels=np.linspace(-4,32,10))

    #cmarker = plt.contour(lon, lat, fff[nmonth[j], :, :], [26], colors='b')
    #plt.clabel(cmarker, fontsize=16, colors=['k'], fmt='%.2f')

    #plt.tight_layout()

cb = plt.colorbar(ax=ax,orientation="horizontal", norm='normlize', pad=0.025, aspect=40, shrink=1)#ax=ax, orientation="horizontal", norm='normlize', pad=0.025, aspect=40, shrink=1
cb.set_label('Temperature', size=3, rotation=90, labelpad=8, fontsize=20)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=15)
'''




'''
fig, axes = plt.subplots(3,3) #nrows=3, ncols=3
ax_list=[]
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        ax_list.append(axes[i,j])
i=0
for ax in ax_list:
    i=i+1
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([180, -180, -90, 90])
    ax.coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
    ax.set_title('风场图', pad=20)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.2, color='black',
                            linestyle='--')
    gl.xlabels_top = False  # 标签
    gl.ylabels_right = False
    gl.xlabels_bottom = True
    gl.ylabels_left = True

    gl.xlines = True  # 网格线
    gl.ylines = True
    # gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
    # gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    ax.contourf(lon, lat, fff[i, :, :], cmap=cp, levels=np.linspace(-4, 32, 10))
plt.show()
'''

'''
fig,ax=plt.subplots(3,3)
ax[0][0] = plt.axes(projection=ccrs.PlateCarree(central_longitude=160))
ax[0][0].set_extent([-180, 180, -90, 90])
ax[0][0].coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
ax[0][0].set_title('海温', pad=20)
ax[0][0].set_xlabel('longitude')
ax[0][0].set_ylabel('latitude')
# Add gridlines
gl = ax[0][0].gridlines(crs=ccrs.PlateCarree(central_longitude=160), draw_labels=False,
                                    linewidth=0.1, color='black',linestyle='--')
gl.xlabels_top = False  # 标签
gl.ylabels_right = False
gl.xlabels_bottom = True
gl.ylabels_left = True

gl.xlines = True  # 网格线
gl.ylines = True
# gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
# gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 20, 'color': 'black'}
gl.ylabel_style = {'size': 20, 'color': 'black'}
k=0
ax[0][0].contourf(lon, lat, fff[k,:,:],cmap=cp,levels=np.linspace(-4,32,10))

ax[0][1] = plt.axes(projection=ccrs.PlateCarree(central_longitude=160))
ax[0][1].set_extent([-180, 180, -90, 90])
ax[0][1].coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
ax[0][1].set_title('海温', pad=20)
ax[0][1].set_xlabel('longitude')
ax[0][1].set_ylabel('latitude')
# Add gridlines
gl = ax[0][1].gridlines(crs=ccrs.PlateCarree(central_longitude=160), draw_labels=False,
                                    linewidth=0.1, color='black',linestyle='--')
gl.xlabels_top = False  # 标签
gl.ylabels_right = False
gl.xlabels_bottom = True
gl.ylabels_left = True

gl.xlines = True  # 网格线
gl.ylines = True
# gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
# gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 20, 'color': 'black'}
gl.ylabel_style = {'size': 20, 'color': 'black'}
k=0
ax[0][1].contourf(lon, lat, fff[k+1,:,:],cmap=cp,levels=np.linspace(-4,32,10))

plt.show()
'''

'''
for i in np.arange(0, 3):
        for j in np.arange(0, 3):
            ax[i][j] = plt.axes(projection=ccrs.PlateCarree(central_longitude=160))
            ax[i][j].set_extent([-180, 180, -90, 90])
            ax[i][j].coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
            ax[i][j].set_title('海温', pad=20)
            ax[i][j].set_xlabel('longitude')
            ax[i][j].set_ylabel('latitude')
            # Add gridlines
            gl = ax[i][j].gridlines(crs=ccrs.PlateCarree(central_longitude=160), draw_labels=False,
                                    linewidth=0.1, color='black',linestyle='--')
            gl.xlabels_top = False  # 标签
            gl.ylabels_right = False
            gl.xlabels_bottom = True
            gl.ylabels_left = True

            gl.xlines = True  # 网格线
            gl.ylines = True
            # gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
            # gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 20, 'color': 'black'}
            gl.ylabel_style = {'size': 20, 'color': 'black'}
            k=i+j
            ax[i][j].contourf(lon, lat, fff[k,:,:],cmap=cp,levels=np.linspace(-4,32,10))

plt.show()
'''

'''
            norm=mpl.colors.Normalize(vmin=-4, vmax=32)
            fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax,
                 orientation='horizontal',extend='both')
'''

