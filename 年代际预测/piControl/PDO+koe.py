import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import cmaps
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import math
import glob
import os
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from eofs.standard import Eof
import cartopy.feature as cfeature
from PyEMD import EEMD
from scipy.signal import hilbert, periodogram
from matplotlib import patches

model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
for i in np.arange(0,1):
    #? 读取数据
    mdl=model[i]
    files=f'F:/data/piControl/{mdl}/tos*.nc'
    combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
    #插值------------------
    target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
    regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                            ignore_degenerate=True,  # 忽略退化的网格点#???某些网格点可能因为几何形状或数值问题导致插值算法无法正常处理
                            )
    regridderdata1= regridder(combined_ds)
    # (-180,180) 转为(0,360)------------------
    regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
    lat_1d = np.array(regridderdata1['lat'].isel(x=0).squeeze().data)  # 提取 NumPy 数组
    lon_1d = np.array(regridderdata1['lon'].isel(y=0).squeeze().data)  # 提取 NumPy 数组
    regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
    ds = regridderdata1.sortby('lon')


    #? 计算PDO： ssta+EOF
    tos_np=ds.tos.isel(time=slice(0,6000),
                    y=slice(110,160),
                    x=slice(120,260))
    # remove the climatological annual cycle ------------------
    shape1=np.array(tos_np.data.shape)
    sst_nps = tos_np.data.reshape((shape1[0]//12,12,shape1[1],shape1[2]))
    print(sst_nps.shape)
    sst_ac = sst_nps.mean(axis=0)  # 直接计算年循环
    sst3 = sst_nps - sst_ac              # 广播减法
    sst3=sst3.reshape((shape1[0], shape1[1], shape1[2])).compute()


    '''#全球变暖趋势
    sst0=ds1.sst.isel(time=slice(0,6000))
    #sst0[sst0==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
    #sst0[sst0==-1.8]=np.nan
    shape=np.array(sst0.data.shape)
    sst_g=sst0.data.reshape((shape[0]//12,12,shape[1],shape[2]))  #1900-2018
    #sst_g[sst_g==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
    #sst_g[sst_g==-1.8]=np.nan
    sst_gm=np.nanmean(sst_g,axis=0)
    sst_gma=sst_g-sst_gm
    global_means = np.nanmean(sst_gma.reshape(shape[0], shape[1], shape[2]), axis=(1,2))
    sst4 = sst3.reshape(shape1[0], shape1[1], shape1[2]) - global_means[:, None, None]
    sst4=sst4.compute()'''

    lat0=tos_np.lat.data
    print(lat0)
    coslat=np.cos(np.deg2rad(lat0))
    weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重
    print(weight)

    solver = Eof(sst3, weights=weight)                        # 创建EOF求解器   #?---
    EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
    PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
    VAR=solver.varianceFraction(neigs=3)

    lon=tos_np.lon.data                        
    lat=tos_np.lat.data
    #print(EOF.shape)

    #EOF[0,:,30]=EOF[0,:,31]                                        #循环补全，相同的效果

    #? koe index: ssta_mean
    sst=ds.tos.isel(
        time=slice(0,6000),
    )#  .data \ .values  .data返回的是xarray数据的底层表示形式，dask\numpy; values, 返回的只是数组
    #print(sst)  #type(sst)  
    # remove the climatological annual cycle-----
    koe = sst.sel(y=slice(120,140),x=slice(145,210))  # (30°–50°N, 145°E–150°W)  40:60,145:210
    print(koe)

    koe_shape=np.array(koe.data.shape)
    yr=koe_shape[0]//12
    print(koe_shape)
    sst_koe=koe.data.reshape((yr, 12, koe_shape[1], koe_shape[2]))
    #sst_koe[sst_koe==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
    #sst_koe[sst_koe==-1.8]=np.nan
    sst_ac = sst_koe.mean(axis=0)                                            # 直接计算年循环
    sst5 = sst_koe - sst_ac                                                         # 广播减法,无需循环，（向量化操作）
    sst5=sst5.reshape((koe_shape[0], koe_shape[1], koe_shape[2]))

    #sst6=sst5-global_means[:, None, None]
    koe_idx=np.nanmean(sst5,axis=(1,2)).compute()     #?---


    #? output -----
    #np.savetxt("D:/decadal prediction/results/piControl/reanalysis/ERSSTv5_PDOindex_1950-2019.txt", PC[:,0], fmt='%1.8f')
    #np.savetxt("D:/decadal prediction/results/piControl/reanalysis/ERSSTv5_PDOpattern_1950-2019.txt", EOF[0,:,:], fmt='%1.8f')
    np.savetxt(f"D:/decadal prediction/results/piControl/{mdl}/koeindex.txt", koe_idx, fmt='%1.8f')


    #? 绘图------------------------------------------------------------------------
    fig=plt.figure(figsize=(18,20),dpi=100) #

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
    plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

    cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
    list_cmap1=cmap1(np.linspace(0,1,20))  
    cp=ListedColormap(list_cmap1,name='cp')

    proj=ccrs.PlateCarree(central_longitude=180)                                            
    ax=fig.add_subplot(211,projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
    contourf_plot=ax.contourf(lon, lat, EOF[0,:,:], cmap=cp,transform=ccrs.PlateCarree(),extend='both', levels=np.linspace(-1, 1, 11))#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.set_extent([120, 260, 20, 70], crs=ccrs.PlateCarree())                                 
    ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
    ax.tick_params(size=6, labelsize=27,pad=12)
    #ax.set_xlabel('longitude',size=18)
    #ax.set_ylabel('latitude',size=18)
    # 画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
    # 但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks([180,-150,120,-120,150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
    ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())

    norm =mpl.colors.Normalize(vmin=-1, vmax=1)
    colorbar = plt.colorbar(contourf_plot, norm=norm, orientation='horizontal', extend='both', shrink=1.0, aspect=37, pad=0.16)
    colorbar.set_label('', fontsize=12)
    colorbar.set_ticks(np.linspace(-1, 1, 11))  # 设置刻度位置 
    colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(-1,1,11)],size=24)
    '''
    norm =mpl.colors.Normalize(vmin=-1, vmax=1)
    colorbar = plt.colorbar(contourf_plot, norm=norm, orientation='horizontal', extend='both', shrink=1, pad=0.03)
    colorbar.set_label('', fontsize=12)
    colorbar.set_ticks(np.linspace(-1, 1, 11))  # 设置刻度位置
    colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(-1,1,11)],size=24)  # 设置刻度标签
    '''
    boxa = patches.Rectangle((145,30 ), 65, 20, linewidth=2, linestyle='--', zorder=2,
                                edgecolor='k', facecolor='none',
                                transform=ccrs.PlateCarree())  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
    ax.add_patch(boxa)


    #? PDO指数和KOE指数的相关性-----------------
    pdo_idx=PC[:,0]
    koe_idx=koe_idx/koe_idx.std()
    r,p=pearsonr(koe_idx,pdo_idx)
    print('r,p',r,p)

    l=len(koe_idx)
    ll=len(koe_idx)-60
    lll=len(koe_idx)-120
    l=1200
    ll=1200-60
    lll=1200-120
    # 5年滑动平均，10年周期
    y1=np.empty((ll))
    y2=np.empty((ll))
    for i in np.arange(30, l-30):
        y1[i-30]=np.mean(koe_idx[i-30: i+30])  
        y2[i-30]=np.mean(pdo_idx[i-30: i+30])  
    r1,p1=pearsonr(y1,y2)  
    print('r1,p1',r1,p1)

    # 10年滑动平均，20年周期
    y3=np.empty((lll))
    y4=np.empty((lll))
    for i in np.arange(60, l-60):
        y3[i-60]=np.mean(koe_idx[i-60: i+60])  
        y4[i-60]=np.mean(pdo_idx[i-60: i+60])  
    r2,p2=pearsonr(y3,y4)  
    print('r2,p2',r2,p2)

    #时间序列-------------------------------------------
    ax1=fig.add_subplot(212)

    x=np.arange(0,l)
    x1=np.arange(0,ll)
    ax1.plot(x,koe_idx[0:l],c='r',lw=0.6,label='koe index')
    ax1.plot(x1+30,y1[0:ll],c='r',lw=1.5,label='5a running mean koe index')
    ax1.plot(x,pdo_idx[0:l],c='b',lw=0.6,label='PDO index')
    ax1.plot(x1+30,y2[0:ll],c='b',lw=1.5,label='5a running mean PDO index')
    ax1.set_xlabel('year',fontsize=27)
    plt.xticks(ticks=np.arange(0,l+1,120),labels=np.arange(0,0+int(l/12)+1,10),size=3,fontsize=27)
    plt.yticks(ticks=np.arange(-4,6),size=3,fontsize=27)
    #ax.tick_params(labelsize=10)
    ax1.annotate(f'r:{r:.2f}  {r1:.2f}  {r2:.2f}',xy=(0.7,0.1),xycoords='axes fraction',c='k',fontsize=27,ha='left',va='top')

    handles,labels = ax1.get_legend_handles_labels()
    ax1.legend(handles,labels, loc='upper center',bbox_to_anchor=(0.5,0.98),ncol=2,
                frameon=False,fontsize=27)

    #子图位置和标题-------------------------------------
    plt.suptitle(f'{mdl}', fontsize=32, x=0.5, y=0.97)# x,y=0.5,0.98(默认)
    plt.subplots_adjust(left=0.08,
                        bottom=0.10,
                        right=0.930,
                        top=0.92,
                        wspace=0.2,      #子图间水平距离
                        hspace=0.1     #子图间垂直距离
                    )
    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/PDO+KOE.png')
    #plt.show()


#? 年变率----------------
'''
x1=np.nanmean(sst_koe,axis=(2,3))
x2=np.nanmean(x1,axis=0)
x3=x1-x2[None,:]
x4=np.sqrt(np.nansum(x3**2,axis=0)/yr)

fig=plt.figure(figsize=(10,8),dpi=100)
x=np.arange(12)
ax1=fig.add_subplot(211)
ax1.plot(x,x2,c='b',label='koe_sst')
ax1.set_xticks(np.arange(0,12))
ax1.set_xlabel('month',size=18)
ax1.set_xticklabels(np.arange(1,13))
ax1.tick_params(size=6, labelsize=18)
plt.legend(fontsize=18,frameon=False)

ax2=fig.add_subplot(212)
ax2.plot(x,x4,c='b',label='koe_std')
ax2.set_xticks(np.arange(0,12))
ax2.set_xlabel('month',size=18)
ax2.set_xticklabels(np.arange(1,13))
ax2.tick_params(size=6, labelsize=18)

plt.suptitle('BCC-CSM2-MR',x=0.5,y=0.98,fontsize=18)
plt.subplots_adjust(left=0.100,
                    bottom=0.150,
                    right=0.970,
                    top=0.92,
                    wspace=0.3,      #子图间水平距离
                    hspace=0.3     #子图间垂直距离
                   )
plt.legend(fontsize=18,frameon=False)
plt.show()
'''