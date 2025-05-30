import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
#from cartopy.util import add_cyclic_point
from cartopy.io.shapereader import Reader
import geopandas
import shapefile
from shapely.geometry import Polygon
a=xr.open_dataset('D:/XC/CHN_gridded_pm25_1979_2019_daily.nc')
#b=xr.open_dataset(('D:/PY/temperature.nc'))
print(a)
#print(b)
a=a.mean(dim='time')
fig=plt.figure()
proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/XC/黑龙江省.shp'
'''
maps=geopandas.read_file('D:/XC/china.shx')
sf=shapefile.Reader(shape_path)
shapes=sf.shapes()
pts=shapes[9].points
prt=shapes[9].parts
codes=[]
x,y=zip(*pts)
fig=plt.figure(figsize=[12,18])
ax=fig.add_subplot(111)
ax.plot(x,y,'_',lw=1,color='k')
plt.show()
'''
#print(maps.area)

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
#数据处理，绘制等值线图
lon=a['lon']
lat=a['lat']
x=a['pm25']
#y=a['jan1963']
lons,lats=np.meshgrid(lon,lat)
data=x
contour_plot=ax.contourf(lons,lats,data)
fig.colorbar(contour_plot)

'''
#格点上标记号
z=y.sel()
w=x.sel()
for i in lon:
    for j in lat:
        if z.sel(longitude=i,latitude=j)-w.sel(longitude=i,latitude=j)>0:
            ax.plot(i,j,linewidth='1',color='m',marker='+')
        else:
            ax.plot(i,j,linewidth='1',color='r',marker='_')

ax.set_title('china 1963.1 temp.contour')
ax_nh.set_title('china nh')
'''
plt.show()
#plt.savefig('D:/PY/result.jpg')
