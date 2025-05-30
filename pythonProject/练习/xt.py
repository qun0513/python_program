import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs     # crs:coordinate reference system
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cmaps
a=xr.open_dataset('E:/test/200_hou.nc')

#a=xr.open_dataset('D:/lianxi/monthaverage.nc')
#a=xr.open_dataset('D:/XC/1979-2021_winter_500hpa_wind .nc')

print(a)
lons=a.longitude
lats=a.latitude
lons=lons[::8]
lats=lats[::8]
#print(lons)
#print(lats)
lon,lat=np.meshgrid(lons,lats)
#u=a.u.mean(dim='time')
#v=a.v.mean(dim='time')
#print(u)
#print(v)
u=a.u[0,::8,::8]
v=a.v[0,::8,::8]
print(u)
print(v)
ws=np.sqrt(u**2+v**2)
print(ws)
ws_d=np.arctan2(u,v)
#
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
#from cartopy.util import add_cyclic_point
from cartopy.io.shapereader import Reader
fig=plt.figure(figsize=(40,25),dpi=150)
proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
#绘制中国地图
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
ax.add_feature(china,lw=1.5,zorder=2)
#grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.set_extent([73,138,3,54])


ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([80,150,0,55])
ax.coastlines(resolution="50m",linewidth=1)  #resolution 分辨率（valid scales）--110m、50m、10m。
#ax.set_title('（d）',pad=20,fontsize=18)
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
gl.xlabel_style = {'size':15, 'color':'black'}
gl.ylabel_style = {'size':15, 'color':'black'}
# Plot windspeed   cmaps.hotres
clevs = np.arange(0,81,2)  #set by max and min of data             #plt.cm.jet  cmaps.sunshine_9lev
plt.contourf(lon, lat, ws, clevs, transform=ccrs.PlateCarree(),cmap=cmaps.hotres,extend='max')
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)
#cb = plt.colorbar(ax=ax, orientation="horizontal", extend='max',pad=0.08, aspect=37, shrink=1,drawedges=True)
#horizontal横vertical竖，shrink收缩比例，ax、cax位置，aspect长宽比，pad距子图，extend两端扩充，extendfrac扩充长度，extendrect扩充形状True，spacing，
#cb.set_label('m/s',size=15,rotation=0,labelpad=1)
#cb.ax.tick_params(labelsize=15)  #参数刻度样式
# Overlay wind vectors
qv = plt.quiver(lon, lat, u, v, width=0.0015,scale=420, color='k')
#width决定箭头箭轴宽度，scale决定箭杆长度  scale=420
#plt.savefig('F:/Rpython/lp28/plot29.1.png',dpi=1200)
#print(u,v)

'''
#绘制南海子图
ax_nh=fig.add_axes([0.540,0.110,0.32,0.25],projection=proj)
ax_nh.add_feature(china,lw=0.5,zorder=2)
ax_nh.add_feature(cfeature.LAND.with_scale('50m'))
ax_nh.add_feature(cfeature.OCEAN.with_scale('50m'))
ax_nh.set_extent([105,125,0,25])
plt.contourf(lon, lat, ws, clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet,extend='both')
qv = plt.quiver(lon, lat, u, v, width=0.0015,scale=420, color='k')
'''

plt.show()