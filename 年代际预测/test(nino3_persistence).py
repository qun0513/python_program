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
import dask
import cftime

ds=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
lon=ds.longitude
lat=ds.latitude

ds['longitude'] = xr.where(ds['longitude'] < 0, ds['longitude'] + 360, ds['longitude'])
ds1 = ds.sortby('longitude')   # 确保经度是递增的
lon1=ds1.longitude
lat1=ds1.latitude

sst0=ds1.sst[0:1788,:,:].data
print(sst0.shape)
sst0[sst0==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
sst0[sst0==-1.8]=np.nan

sst_g=sst0.reshape((149,12,180,360))  #1970-2018
sst_gm=np.nanmean(sst_g,axis=0)
sst_gma=sst_g-sst_gm

tos_np=sst0[:,30:50,145:210]
print(tos_np.shape)
#print(ds1.sst[0:1788,20:70,110:260])

# remove the climatological annual cycle------------------
# 去除年循环（向量化操作）
sst_nps = tos_np.reshape((149, 12, 20, 65))
sst_ac = sst_nps.mean(axis=0)  # 直接计算年循环
sst3 = sst_nps - sst_ac        # 广播减法

global_means = np.nanmean(sst_gma.reshape(1788, 180,360), axis=(1,2))
print(global_means.shape)
sst4 = sst3.reshape(1788, 20, 65) #- global_means[:, None, None]


a = 6357000  
pi = np.pi
lat = ds1.sst[0,30:50,145:210].latitude.data

dx = 2 * pi * a * np.cos(np.deg2rad(lat)) / 360 
dy = 2 * pi * a / 360  #纬度按1度规律递减时

s_grid = (dx[:, np.newaxis] * dy) * np.ones(( 65,))     #-----
valid_mask = ~np.isnan(sst4[0])  
print(valid_mask)
total_area = np.sum(s_grid * valid_mask)  #在布尔型数组与数值型数组相乘时，True会被当作1，False会被当作0.
weight = s_grid / total_area
sst = np.nansum(sst4 * weight[np.newaxis, :, :], axis=(1, 2))


# lag autocoorelation---------------------------------------------------------
lag_ar=np.empty((12,12))
p=np.empty((12,12))
for i in np.arange(0,12):
    #print(i)
    for j in np.arange(0,12):
        if i+j+1>11:   #11*n
            lag_ar[i,j], p[i,j]=pearsonr(sst[i:148*12:12], sst[i+j+1:149*12:12])  #499\500, 68\69
        else:
            lag_ar[i,j], p[i,j]=pearsonr(sst[i:148*12:12], sst[i+j+1:148*12:12])
    #print(i)
'''
# 预生成索引数组
n_years = 36
months = 12
total_samples = n_years * months

# 初始化结果数组
lag_ar = np.empty((12, 12))
p = np.empty((12, 12))

for lead in range(12):
    for month in range(12):
        start1 = month
        end1 = total_samples - (lead + 1)
        start2 = month + lead + 1
        end2 = total_samples
        
        x = sst[start1:end1:12]
        y = sst[start2:end2:12]
        
        lag_ar[month, lead], p[month, lead] = pearsonr(x, y)
    print(lead)
'''
#print('lag_ar:',lag_ar)
#print('p:', p)

# plot ---------------------------------------------------------------------------------
x=np.arange(0,12)
y=np.arange(0,12)
X,Y=np.meshgrid(x,y)
cmap1=cmaps.BlueDarkRed18                                        # BlueDarkRed18 temp_diff_18lev MPL_YlOrBr
list_cmap1=cmap1(np.linspace(0,1,20))
cp=ListedColormap(list_cmap1,name='cp')

fig=plt.figure(figsize=(10,8), dpi=100)
ax=fig.add_subplot()
contour_plot=ax.contour(X, Y, lag_ar, colors='k', levels=np.linspace(-1, 1, 21), linestyles='solid')
plt.contourf(X, Y, p, levels=[0, 0.05, 1], colors=['yellow', 'none']) 
plt.tick_params(labelsize=12, size=3)
plt.clabel(contour_plot, contour_plot.levels, inline=True, fontsize=10, fmt='%0.1f')#,manual=[(i,4) for i in np.arange(0,12)])
plt.xticks(ticks=np.arange(0,12,1), labels=np.arange(1,13), size=25)
plt.yticks(ticks=np.arange(0,12,1), labels=np.arange(1,13), size=25)
ax.set_xlabel('lead time',size=26)
ax.set_ylabel('month',size=26)

plt.subplots_adjust(left=0.10,
                    bottom=0.100,
                    right=0.900,
                    top=0.900,
                    wspace=0.18,                 # horizontal distance between subgraphs
                    hspace=0.300                # vertical distance between subgraphs
                )

plt.suptitle('koe_persistence_HadISST_1870-2018', fontsize=30, x=0.5, y=0.96)   # x,y=0.5,0.98 (default)
plt.savefig("D:/decadal prediction/results/piControl/koe_persistence_HadISST_1870-2018.png")

plt.show()