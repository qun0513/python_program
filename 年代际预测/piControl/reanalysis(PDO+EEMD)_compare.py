 #? 改进的去除年循环、去除全球变暖趋势

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

ds=xr.open_dataset("D:/pycharm/HadISST_sst_187001-202501.nc")
lon=ds.longitude
lat=ds.latitude

ds['longitude'] = xr.where(ds['longitude'] < 0, ds['longitude'] + 360, ds['longitude'])
ds1 = ds.sortby('longitude')   # 确保经度是递增的

sst0=ds1.sst.sel(time=slice('1980-01-01','2020-01-01'))
#sst0[sst0==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
#sst0[sst0==-1.8]=np.nan
shape=np.array(sst0.data.shape)
sst_g=sst0.data.reshape((shape[0]//12,12,shape[1],shape[2]))  #1900-2018
sst_g[sst_g==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
sst_g[sst_g==-1.8]=np.nan
sst_gm=np.nanmean(sst_g,axis=0)
sst_gma=sst_g-sst_gm


tos_np=ds1.sst.sel(time=slice('1980-01-01','2020-01-01'),
                   latitude=slice(70,20),
                   longitude=slice(120,260))
# remove the climatological annual cycle ------------------
# 去除年循环（向量化操作）
shape1=np.array(tos_np.data.shape)
sst_nps = tos_np.data.reshape((shape1[0]//12,12,shape1[1],shape1[2]))
sst_nps[sst_nps==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
sst_nps[sst_nps==-1.8]=np.nan
sst_ac = sst_nps.mean(axis=0)  # 直接计算年循环
sst3 = sst_nps - sst_ac              # 广播减法

global_means = np.nanmean(sst_gma.reshape(shape[0], shape[1], shape[2]), axis=(1,2))
sst4 = sst3.reshape(shape1[0], shape1[1], shape1[2]) - global_means[:, None, None]

lat0=tos_np.latitude.data
print(lat0)
coslat=np.cos(np.deg2rad(lat0))
weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重
print(weight)
solver = Eof(sst4, weights=weight)                        # 创建EOF求解器   #?---
EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
VAR=solver.varianceFraction(neigs=3)

lon=tos_np.longitude.data                        
lat=tos_np.latitude.data


#EOF[0,:,30]=EOF[0,:,31]                                        #循环补全，相同的效果

#? koe index------------------------<
sst=ds1.sst.sel(
    time=slice('1980-01-01','2020-01-01'),
)#  .data \ .values  .data返回的是xarray数据的底层表示形式，dask\numpy; values, 返回的只是数组
#print(sst)  #type(sst)  

# remove the climatological annual cycle
koe = sst.sel(latitude=slice(50,30),longitude=slice(145,210))  # (30°–50°N, 145°E–150°W)  40:60,145:210
print(koe)
koe_shape=np.array(koe.data.shape)
yr=koe_shape[0]//12
print(koe_shape)
sst_koe=koe.data.reshape((yr, 12, koe_shape[1], koe_shape[2]))#.compute()
sst_koe[sst_koe==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
sst_koe[sst_koe==-1.8]=np.nan
sst_ac = sst_koe.mean(axis=0)                                            # 直接计算年循环
sst5 = sst_koe - sst_ac                                                         # 广播减法,无需循环，（向量化操作）
sst5=sst5.reshape(koe_shape[0], koe_shape[1], koe_shape[2])
sst6=sst5-global_means[:, None, None]
koe_idx=np.nanmean(sst6,axis=(1,2))     #?---

#? output -----
#np.savetxt("D:/decadal prediction/results/piControl/reanalysis/ERSSTv5_PDOindex_1950-2019.txt", PC[:,0], fmt='%1.8f')
#np.savetxt("D:/decadal prediction/results/piControl/reanalysis/ERSSTv5_PDOpattern_1950-2019.txt", EOF[0,:,:], fmt='%1.8f')
#np.savetxt("D:/decadal prediction/results/piControl/reanalysis/ERSSTv5_koeindex_1950-2019.txt", koe_idx, fmt='%1.8f')


#? 绘图------------------------------------------------------------------------
'''fig=plt.figure(figsize=(8,10),dpi=100) #

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
# 画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
# 但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.

ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([180,-150,120,-120,150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())

boxa = patches.Rectangle((145,30), 65, 20, linewidth=2, linestyle='--', zorder=2,
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

#时间序列-------------------------------------------
ax1=fig.add_subplot(212)

x=np.arange(0,l)
x1=np.arange(0,ll)
ax1.plot(x,koe_idx[0:l],c='r',lw=0.6,label='koe index')
ax1.plot(x1+30,y1[0:ll],c='r',lw=1.5,label='5a running mean koe index')
ax1.plot(x,pdo_idx[0:l],c='b',lw=0.6,label='PDO index')
ax1.plot(x1+30,y2[0:ll],c='b',lw=1.5,label='5a running mean PDO index')
ax1.set_xlabel('year',fontsize=20)
plt.xticks(ticks=np.arange(0,l+1,120),labels=np.arange(1950,1950+int(l/12)+1,10),size=3,fontsize=20)
plt.yticks(ticks=np.arange(-4,6),size=3,fontsize=20)
#ax.tick_params(labelsize=10)
#ax1.annotate(f'r:{r:.2f}',xy=(0.7,0.1),xycoords='axes fraction',c='k',fontsize=18,ha='left',va='top')

handles,labels = ax1.get_legend_handles_labels()
ax1.legend(handles,labels, loc='upper center',bbox_to_anchor=(0.5,1.05),ncol=2,
            frameon=False,fontsize=17)

#子图位置和标题-------------------------------------
plt.suptitle('ERSSTv5', fontsize=20, x=0.5, y=0.98)# x,y=0.5,0.98(默认)

plt.show()'''

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

plt.suptitle('HadISST_1980-2019',x=0.5,y=0.98,fontsize=18)
plt.subplots_adjust(left=0.100,
                    bottom=0.150,
                    right=0.970,
                    top=0.92,
                    wspace=0.3,      #子图间水平距离
                    hspace=0.3     #子图间垂直距离
                   )
plt.legend(fontsize=18,frameon=False)
plt.show()

#? EEMD -----
'''eemd = EEMD(trials = 400,
                        noise_width=0.2,
                        random_seed=None,
                        sift_threshold=0.2,                     # 宽松的SD阈值适应低频信号
                        S_number=6,                             # 严格筛选低频IMF
                        spline_kind='cubic',                   # 更光滑的插值方法:cubic\akima
                        extrema_detection='parabol',   # 镜像延拓处理边界
                        #n_processes=4                        # 并行加速
                        )                      

l=len(koe_idx)
x=np.arange(0,l)
y=koe_idx
#y=PC[0:l,0]
#x1=np.arange(0,5940)
#y1=np.empty((5940))
#for i in np.arange(30, len(y)-30):
#    y1[i-30]=np.mean(y[i-30: i+30])

IMFs = eemd.eemd(y,x)     # signal, time
n_imfs = len(IMFs)

# 1. Calculate the IMF cycle using the Hilbert transform method------------------
def calculate_imf_period(imf, dt=1/12):
    analytic_signal = hilbert(imf)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi * dt)
    
    # 剔除异常值：仅保留0.01-1 Hz（1-100年周期）
    mask = (instantaneous_frequency > 0.01) & (instantaneous_frequency < 1)
    if np.sum(mask) < 10:  # 有效数据不足时返回NaN
        return np.nan
    valid_freq = instantaneous_frequency[mask]
    return 1 / np.mean(valid_freq)

periods = [calculate_imf_period(IMFs[i]) for i in range(n_imfs)]
print('periods:',periods)

# 1.1  counting cross zero numbers
tt=[]
for j in np.arange(0,len(IMFs)):
    imfs=IMFs[j]
    kk=0
    for i in np.arange(0,len(imfs)-1):
        if imfs[i]*imfs[i+1]<0:
            kk=kk+1
    t=len(imfs)*2/kk/12
    #print('t:', len(imfs)*2/kk)
    tt.append(t)   #periods
print('tt: ',tt,)

"""kkk=0
for i in np.arange(0,len(y1)-1):
    if y1[i]*y1[i+1]<0:
        kkk=kkk+1
ttt=len(y1)*2/kkk/12
print(ttt)"""

# 2. calculate explained variance ------------------
def calculate_variance_contribution(imfs):
    svar = []
    for imf in imfs:
        var = np.var(imf)
        svar.append(var)
    var_contributions=svar/sum(svar)*100
    return var_contributions
var_contrib = calculate_variance_contribution(IMFs)
print('var_contrib:', var_contrib, sum(var_contrib[:]))

# plot -----------------------------------------------------------------------------------
fig=plt.figure(figsize=(8, 10),dpi=100)


ax=fig.add_subplot(n_imfs + 1, 1, 1)
ax.plot(x, y, color='r', linewidth=0.6)     # original signal
#ax.plot(x1+30, y1, color='k', linewidth=1)     # original signal
ax.set_xticks([])
ax.tick_params(labelsize=9, size=3)
#ax.annotate(f"{ttt:.2f}yr",xy=(1.01,0.3),xycoords='axes fraction',c='k',fontsize=9)


for i, imf in enumerate(IMFs):
    ax1=fig.add_subplot(n_imfs + 1, 1, i + 1)
    ax1.plot(x, imf, color='b', linewidth=0.8)     # each IMF
    ax1.set_xticks([])
    ax1.tick_params(labelsize=9, size=3)
    ax2=ax1.twinx()
    ax2.set_yticks([])
    ax2.annotate(f"{tt[i]:.2f}yr\n{var_contrib[i]:.2f}",xy=(1.01,0.3),xycoords='axes fraction',c='k',fontsize=9)#,ha='left',va='top')
    #ax2.set_ylabel(f"{tt[i]:.2f}yr\n{var_contrib[i]:.2f}" , size=9, labelpad=45, rotation=0, loc='top')
plt.xticks(ticks=np.arange(0, 6001, 600), labels=[f"{i}" for i in np.arange(0, 501, 50)], size=9)
ax1.set_xlabel('year', size=9)

plt.show()'''

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
'''


































