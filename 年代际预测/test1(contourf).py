import numpy as np 
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps

a=xr.open_dataset('D:/decadal prediction/data/tos_Omon_MIROC6_dcppA-assim_r10i1p1f1_gn_195001-201712.nc',decode_times=False)
#a['time']=pd.date_range(start='1/2015',end='1/2051',freq='m')  #更改时间维的命名方式  end='1/2051'  periods=432
print(a.tas)
a1=a.tas.mean(dim='time')

#print(a.tas.data[:,45:135,::2])
#print(a.tas.data[:,45:135,::2].shape)

lon=a.lon
lat=a.lat
from cartopy.util import add_cyclic_point
a1,lon=add_cyclic_point(a1,coord=lon)

proj=ccrs.PlateCarree()#central_longitude=180
fig=plt.figure()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cmap1=cmaps.BlueDarkRed18#BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

ax=fig.add_subplot(projection=proj)#central_longitude=180)           #之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lon, lat,a1, cmap=cmap1)   #, levels=np.linspace(-4, 32, 10)
ax.set_extent([-180, 180, 0, 360],crs=proj)
ax.coastlines(resolution="50m", linewidth=1)  # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('longitude',size=18)
ax.set_ylabel('latitude',size=18)

ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([-180,120,-60,0,60,-120,180],crs=proj)   #[ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(-90,91,30), crs=proj)

plt.show()