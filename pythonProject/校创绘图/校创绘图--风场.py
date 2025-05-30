import numpy as np
import pandas as pd
import xarray as xr
#在地图上绘制风场和轨迹
import matplotlib.pyplot as plt###引入库包
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import netCDF4 as nc
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = 'Arial'  #默认字体类型
mpl.rcParams["mathtext.fontset"] = 'cm' #数学文字字体
mpl.rcParams["font.size"] = 15   #字体大小
mpl.rcParams["axes.linewidth"] = 1   #轴线边框粗细（默认的太粗了）

f1=xr.open_dataset('D:/XC/1979-2021_winter_500hpa_wind .nc')
f2=xr.open_dataset('D:/XC/wind-big-500.nc')
#f1=nc.Dataset('wind.nc') #打开.nc文件
#读取文件中的数据
print(f1)
print(f2)
lat=f1.variables['latitude'][:]
lon=f1.variables['longitude'][:]
time=f1.variables['time'][:]
u1000=f1.variables['u'][:]#下载的1000hPa的风场数据
v1000=f1.variables['v'][:]

lon2d, lat2d = np.meshgrid(lon, lat)
u1000_aim=np.mean(u1000[:,:,:],axis=0)
v1000_aim=np.mean(v1000[:,:,:],axis=0)

proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(10,8),dpi=550)  # 创建画布
ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  # 创建子图


u_all=u1000_aim
v_all=v1000_aim

#-----------绘制地图-------------------------------------------

# ax.add_feature(cfeature.LAND.with_scale('50m'))####添加陆地######
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))#####添加海岸线#########
# ax.add_feature(cfeature.OCEAN.with_scale('50m'))######添加海洋########

#-----------添加经纬度---------------------------------------
extent=[-30,150,0,90]##经纬度范围
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0., color='k', alpha=0.5, linestyle='--')
dlon, dlat = 45, 30   #设置步长
xticks = np.arange(0, 360.1, dlon)  #设置绘图范围
yticks = np.arange(-90, 90.1, dlat)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())  #图幅设置坐标轴刻度
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))  #设置坐标轴刻度标签格式
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent(extent)     #显示所选择的区域

#----------修改国界，并添加省界-----------------------------
#在这个网站上可以找到dat文件，https://gmt-china.org/data/
# with open('C:/Users/hj/.local/share/cartopy/shapefiles/natural_earth/physical/CN-border-La.dat') as src:
#     context = src.read()
#     blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
#     borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
# for line in borders:
#     ax.plot(line[0::2], line[1::2], '-', color='k',lw=0.3, transform=ccrs.Geodetic())

#-------------------plot---------------------------
#levels = np.arange(1004, 1032 + 1, 1)
#cb = ax.contourf(lon2d,lat2d,msl_all, levels=levels, cmap='Spectral_r',transform=ccrs.PlateCarree())

cq =ax.quiver(lon2d[::20,::20],lat2d[::20,::20],u_all[::20,::20],v_all[::20,::20],color='k',scale=250,zorder=10,width=0.002,headwidth=3,headlength=4.5,transform=ccrs.PlateCarree())

plt.show()
