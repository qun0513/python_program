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
import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import cmaps
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

'''
ds=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
lon=ds.longitude
lat=ds.latitude
print(ds)
print(ds.sst[0,50,50].data)

#? 可行的-----------------------------------------------------------------------------------------------
ds['longitude'] = xr.where(ds['longitude'] < 0, ds['longitude'] + 360, ds['longitude'])
ds1 = ds.sortby('longitude')   # 确保经度是递增的
lon1=ds1.longitude
lat1=ds1.latitude
print(ds1.sst[0,50,50].data)
#? -------------------------------------------------------------------------------------------------------
'''


combined_ds=xr.open_dataset("F:/data/piControl/HadGEM3-GC31-MM(LL)/tos_Omon_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn_185001-186912.nc")
target_grid = xe.util.grid_global(1, 1)
regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                         #, filename=None,
                         #periodic=True,                 # 处理经度周期性（如全球网格）
                         ignore_degenerate=True,  # 忽略退化的网格点#???某些网格点可能因为几何形状或数值问题导致插值算法无法正常处理
                         #extrap_method='nearest_s2d'  # 外推方法
                         )
regridderdata1= regridder(combined_ds)
print('regridderdata1',regridderdata1.tos[0,50,50].data)

# 调整经度范围
regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
# 提取一维经纬度的数值
lat_1d = regridderdata1['lat'].isel(x=0).squeeze().data  # 提取 NumPy 数组
lon_1d = regridderdata1['lon'].isel(y=0).squeeze().data  # 提取 NumPy 数组
# 重新分配坐标
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))  #lat的索引值是y
print(regridderdata1.lat.dims)
# 按经度排序
ds = regridderdata1.sortby('lon')
print('ds',ds)

#north_pacific = ds.sel(y=slice(110,160), x=slice(110, 260))  #
#north_pacific_subset = north_pacific.isel(time=slice(0, 100))  #n=100
ds_tos=ds.tos[0:1,110:160,110:260]
lon=ds.lon[110:260]
lat=ds.lat[110:160]

#绘图------------------------------------
#? optimized -- north pacific area  optimized
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']         # display Chinese labels normally: SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False                     # display negative sign normally
cmap1=cmaps.BlueDarkRed18                                        # BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,20))
cp=ListedColormap(list_cmap1,name='cp')

proj=ccrs.PlateCarree(central_longitude=180 )    #central_longitude=180
fig=plt.figure(figsize=(10,12),dpi=100)  # x*y                     
ax=fig.add_subplot(2,1,1,projection=proj)                               
ax.contourf( lon, lat, ds_tos[0, :, :] , cmap=cp, levels=np.linspace(-30, 30, 11), transform=ccrs.PlateCarree() )   #, levels=np.linspace(-4, 32, 10)
ax.set_extent([120, 240, 20, 60], crs=ccrs.PlateCarree())       
ax.coastlines(resolution="50m", linewidth=0.8)     
ax.set_title('spatial mode',size=15, pad=10)
ax.tick_params(size=3, labelsize=12)
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
#ax.set_xticks([180,150,120,-120,-150,-180],crs=ccrs.PlateCarree(central_longitude=180))  
ax.set_xticks([120, 150, 180, 210, 240], crs=ccrs.PlateCarree())    
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())
#ax.set_xlabel('longitude',size=12)
#ax.set_ylabel('latitude',size=12)

#Subgraphs position and title-------------------------------------
plt.subplots_adjust(left=0.070,
                    bottom=0.066,
                    right=0.950,
                    top=0.890,
                    wspace=0.18,                 # horizontal distance between subgraphs
                    hspace=0.300                # vertical distance between subgraphs
                )
#plt.tight_layout()
#plt.suptitle('PDO of piControl for NorCPM1', fontsize=18, x=0.5, y=0.98)   # x,y=0.5,0.98 (default)
#plt.savefig("D:/decadal prediction/results/piControl/PDO of piControl for NorCPM1.png")
plt.show()
