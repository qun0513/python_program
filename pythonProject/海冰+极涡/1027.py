import xarray as xr
import matplotlib             #解决Linux无法可视化的问题
#matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray import DataArray

#数据读取————————————————————————————————————————————————————————————————————————————————————————
#SIC=xr.open_dataset('/home/dell/ZQ19/HadISST_ice_187001-202102.nc')
SIC=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
sic=SIC.sic.loc[:,89.5:39.5,120.5:-80.5]
sic=sic.isel(time=slice(1308,1796))          #1979-2020 (1979.1-2019.08)

sic=sic.mean(dim='time')

lon=sic.longitude
lat=sic.latitude

lon,lat=np.meshgrid(lon,lat)

plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
fig = plt.figure(figsize=(40,25),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([120,-80,0,90])
ax.coastlines(resolution="50m",linewidth=1)  #resolution 分辨率（valid scales）--110m、50m、10m。
ax.set_title('风场图',pad=20)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,linewidth=0.2, color='black', linestyle='--')
gl.xlabels_top = False       #标签
gl.ylabels_right = False
gl.xlabels_bottom= True
gl.ylabels_left=True

gl.xlines = True           #网格线
gl.ylines=True
#gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
#gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':10, 'color':'black'}
gl.ylabel_style = {'size':10, 'color':'black'}
# Plot windspeed
#clevs = np.arange(-30,41,2)  #set by max and min of data
plt.contourf(lon, lat, sic,  transform=ccrs.PlateCarree(),cmap=plt.cm.jet,extend='both')
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)
cb = plt.colorbar(ax=ax, orientation="vertical", extend='both',pad=0.04, aspect=20, shrink=0.8,drawedges=True)
#horizontal横vertical竖，shrink收缩比例，ax、cax位置，aspect长宽比，pad距子图，extend两端扩充，extendfrac扩充长度，extendrect扩充形状True，spacing，
cb.set_label('m/s',size=11,rotation=0,labelpad=10)
cb.ax.tick_params(labelsize=7)  #参数刻度样式
# Overlay wind vectors

plt.show()

