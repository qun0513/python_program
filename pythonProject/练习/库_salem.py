import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth
#from xarray.backends import NetCDF4DataStoreg
import salem
from datetime import datetime
#from siphon.catalog import TDSCatalog

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas

a=xr.open_dataset('D:/XC/wind-big-850.nc')
lon=a.longitude
lat=a.latitude
lon,lat=np.meshgrid(lon,lat)
a=a.mean(dim='time')

'''
def create_map():
    shp_path = './cn_shp/Province_9/'
    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(6, 8), dpi=400)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    # --设置地图属性
    provinces = cfeat.ShapelyFeature(
        Reader(shp_path + 'Province_9.shp').geometries(),
        proj, edgecolor='k',
        facecolor='none'
    )
    # 加载省界线
    ax.add_feature(provinces, linewidth=0.6, zorder=2)
    # 加载分辨率为50的海岸线
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    # 加载分辨率为50的河流~
    ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)
     # 加载分辨率为50的湖泊
    ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)
     # --设置网格属性
    gl = ax.gridlines(
        crs = ccrs.PlateCarree(),
        draw_labels = False,
        linewidth = 0.9,
        color = 'k',
        alpha = 0.5,
        linestyle = '--'
    )
    gl.xlabels_top = gl.ylabels_right = gl.ylabels_left = gl.ylabels_bottom = False  # 关闭经纬度标签
    # --设置刻度
    ax.set_xticks(np.arange(90, 145 + 5, 5))
    ax.set_yticks(np.arange(0, 50 + 5, 5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=5, direction='out')
    # -- 设置范围
    ax.set_extent([90, 140, 0, 50], crs=ccrs.PlateCarree())
    return ax

# -- 获取数据
best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                    'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')
ncss = best_gfs.datasets[0].subset()
query = ncss.query()
query.lonlat_box(north=50, south=0, east=150, west=90).time(datetime.utcnow())
query.variables('Temperature_surface')
query.accept('netcdf4')
nc = ncss.get_data(query)
data = xr.open_dataset(NetCDF4DataStore(nc))
temp = data['Temperature_surface'].isel(time=0)

# --设置colorbar
cbar_kwargs = {
    'orientation': 'horizontal',
    'label': 'Temp',
    'shrink': 0.8,
}

# --设置level
levels = np.arange(270, 310, 1)

# -- 画图
temp.plot.contourf(
    ax=create_map(),
    cmap='Spectral_r',
    levels=levels,
    cbar_kwargs=cbar_kwargs,
    transform=ccrs.PlateCarree(),
    extend='both'
)
plt.savefig('temp.png')

# -- 读取陆地shp，并使用salem.roi来提取感兴趣的区域。
shp_path = './ne_10m_land_scale_rank/'
shp = geopandas.read_file(shp_path + 'ne_10m_land_scale_rank.shp')
t = temp.salem.roi(shape=shp)
t.plot.contourf(
    ax=create_map(),
    cmap='Spectral_r',
    levels=levels,
    cbar_kwargs=cbar_kwargs,
    transform=ccrs.PlateCarree()
)
plt.savefig('temp_land.png')

# -- 读取海洋shp，并使用salem.roi来提取感兴趣的区域。
shp_path = './ne_10m_ocean_scale_rank/'
shp = geopandas.read_file(shp_path + 'ne_10m_ocean_scale_rank.shp')
t = temp.salem.roi(shape=shp)
t.plot.contourf(
    ax=create_map(),
    cmap='Spectral_r',
    levels=levels,
    cbar_kwargs=cbar_kwargs,
    transform=ccrs.PlateCarree()
)
plt.savefig('temp_ocean.png')
'''

# -- 读取中国各省shp，并使用salem.roi来提取感兴趣的区域。

shp_path = 'D:/XC/China_shp_lqy/'
shp = geopandas.read_file(shp_path + 'province.shp')
criterion = (shp['省']=='广东省') | (shp['省']=='湖南省') | (shp['省']=='福建省') | (shp['省']=='江西省')
shp = shp[criterion]
t = a.salem.roi(shape=shp)
print(t)


import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

fig,ax= plt.subplots(figsize=(50,45),dpi=100)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-30,150,0,90])
ax.coastlines(resolution="50m",linewidth=1)  #resolution:110m\50m\10m

#绘制中国地图
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
ax.add_feature(china,lw=0.5,zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'))  #scale:110m\50m\10m
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.tick_params(labelsize=16)

clevs = np.linspace(-10,10,21)
plt.contourf(lon, lat, t, clevs, transform=ccrs.PlateCarree(),cmap=mpl.cm.jet,extend='both')
#MPL_PiYG_r
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('m/s',size=14,rotation=0,labelpad=15,fontsize=20)
cb.ax.tick_params(labelsize=10)

plt.title('',size=20,pad=20)

plt.show()


