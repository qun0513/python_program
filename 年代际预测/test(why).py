import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import cmaps
import matplotlib as mpl
from matplotlib.colors import ListedColormap

files = 'F:/data/piControl/MIROC6/tos*.nc' 
combined_ds = xr.open_mfdataset(
    paths=files, 
    use_cftime=True
)

#插值------------------
target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                         ignore_degenerate=True                         )
regridderdata1= regridder(combined_ds)

regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])

lat_1d = regridderdata1['lat'].isel(x=0).squeeze().data  # 提取 NumPy 数组
lon_1d = regridderdata1['lon'].isel(y=0).squeeze().data  # 提取 NumPy 数组
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
ds = regridderdata1.sortby('lon')
print('------------------------1')
tos_koe=ds.tos[0:6000,120:140,145:210].data

sst_nps = tos_koe.reshape((500, 12, 20, 65))
sst_ac = sst_nps.mean(axis=0)  # 直接计算年循环
sst3 = sst_nps - sst_ac        # 广播减法
sst4 = sst3.reshape((6000, 20, 65))
print('----------------2')

import numpy as np
a = 6357000  
pi = np.pi
lat = np.arange(30, 50)  
dx = 2 * pi * a * np.cos(np.deg2rad(lat)) / 360  
dy = 2 * pi * a / 360  
s_grid = (dx[:, np.newaxis] * dy) * np.ones((65,))  
valid_mask = ~np.isnan(sst4[0]) 
total_area = np.sum(s_grid * valid_mask)  # 单位：平方米
weight = s_grid / total_area  # 每个格点的权重系数
sst = np.nansum(sst4 * weight[np.newaxis, :, :], axis=(1, 2))  # 结果形状: (6000,)
print(sst)
print('--------4')

#sst=np.nanmean(np.nanmean((sst10),axis=1),axis=1) #sst4.mean() or sst(weight area); sst10(8,9)


# lag autocoorelation---------------------------------------------------------
lag_ar=np.empty((12,12))
p=np.empty((12,12))
for i in np.arange(0,12):
    print(i)
    for j in np.arange(0,12):
        if i+j+1>11:   #11*n
            lag_ar[i,j], p[i,j]=pearsonr(sst[i:499*12:12], sst[i+j+1:500*12:12])  #499\500, 68\69
        else:
            lag_ar[i,j], p[i,j]=pearsonr(sst[i:499*12:12], sst[i+j+1:499*12:12])
    print(i)

print('lag_ar:',lag_ar)
print('p:', p)

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

plt.suptitle('koe_persistence_MIROC6_500', fontsize=30, x=0.5, y=0.96)   # x,y=0.5,0.98 (default)
plt.savefig("D:/decadal prediction/results/piControl/koe_persistence_MIROC6_500.png")

plt.show()