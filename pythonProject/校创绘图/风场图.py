import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
a=xr.open_dataset('D:/XC/wind-big-500.nc')
#a=xr.open_dataset('D:/XC/1979-2021_winter_500hpa_wind .nc')
print(a)
lons=a.longitude
lats=a.latitude
lons=lons[::10]
lats=lats[::10]
print(lons)
print(lats)
lon,lat=np.meshgrid(lons,lats)
u=a.u.mean(dim='time')
v=a.v.mean(dim='time')
print(u)
print(v)
u=u[::10,::10]
v=v[::10,::10]
#print(u)
#print(v)
ws=np.sqrt(u**2+v**2)
ws_d=np.arctan2(u,v)
#
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
#from cartopy.util import add_cyclic_point
from cartopy.io.shapereader import Reader
fig = plt.figure(figsize=(40,25),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
proj=ccrs.PlateCarree()

shape_path='D:/PY/china-myclass.shp'
#绘制中国地图
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
ax.add_feature(china,lw=0.5,zorder=2)
grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.set_extent([73,138,3,54])
#绘制南海子图
ax_nh=fig.add_axes([0.61,0.128,0.32,0.25],projection=proj)
ax_nh.add_feature(china,lw=0.5,zorder=2)
ax_nh.add_feature(cfeature.LAND.with_scale('50m'))
ax_nh.add_feature(cfeature.OCEAN.with_scale('50m'))
ax_nh.set_extent([105,125,0,25])

ax.set_extent([-30,150,0,90])
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
clevs = np.arange(-30,41,2)  #set by max and min of data
plt.contourf(lon, lat, ws[:,:], clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet,extend='both')
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)
cb = plt.colorbar(ax=ax, orientation="vertical", extend='both',pad=0.04, aspect=20, shrink=0.8,drawedges=True)
#horizontal横vertical竖，shrink收缩比例，ax、cax位置，aspect长宽比，pad距子图，extend两端扩充，extendfrac扩充长度，extendrect扩充形状True，spacing，
cb.set_label('m/s',size=11,rotation=0,labelpad=10)
cb.ax.tick_params(labelsize=7)  #参数刻度样式
# Overlay wind vectors
qv = plt.quiver(lon, lat, u[:,:], v[:,:], width=0.0006,scale=920, color='k')
#width决定箭头箭轴宽度，scale决定箭杆长度  scale=420
#plt.savefig('F:/Rpython/lp28/plot29.1.png',dpi=1200)
print(u,v)
plt.show()

'''
from netCDF4 import Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

data=Dataset('D:/XC/1979-2021_winter_500hpa_wind .nc')
# Run the following cell to see the MERRA2 metadata. This line will print attribute and variable information. From the 'variables(dimensions)' list, choose which variable(s) to read in below:
print(data)
# Read in variables:
# longitude and latitude
lons = data.variables['longitude']
lons=lons[::10]
print(lons)
lats = data.variables['latitude']
lats=lats[::10]
lon, lat = np.meshgrid(lons, lats)
# 2-meter eastward wind m/s
U2M = data.variables['u']
# 2-meter northward wind m/s
V2M = data.variables['v']
# Replace _FillValues with NaNs:
U2M_nans = U2M[:]
V2M_nans = V2M[:]
_FillValueU2M = U2M._FillValue
_FillValueV2M = V2M._FillValue
U2M_nans[U2M_nans == _FillValueU2M] = np.nan
V2M_nans[V2M_nans == _FillValueV2M] = np.nan
# Calculate wind speed:
ws = np.sqrt(U2M_nans**2+V2M_nans**2)
#print(ws)
# Calculate wind direction in radians:
ws_direction = np.arctan2(V2M_nans,U2M_nans)
# NOTE: the MERRA-2 file contains hourly data for 24 hours (t=24). To get the daily mean wind speed, take the average of the hourly wind speeds:
ws_daily_avg = np.nanmean(ws, axis=0)
# NOTE: To calculate the average wind direction correctly it is important to use the 'vector average' as atan2(<v>,<u>) where <v> and <u> are the daily average component vectors, rather than as mean of the individual wind vector direction angle.  This avoids a situation where averaging 1 and 359 = 180 rather than the desired 0.
U2M_daily_avg = np.nanmean(U2M_nans, axis=0)
V2M_daily_avg = np.nanmean(V2M_nans, axis=0)
ws_daily_avg_direction = np.arctan2(V2M_daily_avg, U2M_daily_avg)

fig = plt.figure(figsize=(4000,2500),dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([70,80,0,5])
ax.coastlines(resolution="50m",linewidth=1)
# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True
gl.xlocator = mticker.FixedLocator([-65,-60,-50,-40,-30])
gl.ylocator = mticker.FixedLocator([30,40,50,60])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':10, 'color':'black'}
gl.ylabel_style = {'size':10, 'color':'black'}
# Plot windspeed
clevs = np.arange(-30,41,1)
plt.contourf(lon, lat, ws[0,:,:], clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
#cb.set_label('m/s',size=14,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=10)
# Overlay wind vectors
qv = plt.quiver(lon, lat, U2M_nans[0,:,:], V2M_nans[0,:,:], width=0.0015,scale=320, color='k')
#width决定箭头箭轴宽度，scale决定箭杆长度  scale=420
#plt.savefig('F:/Rpython/lp28/plot29.1.png',dpi=1200)
plt.show()
'''

