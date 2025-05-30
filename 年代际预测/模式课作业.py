import xesmf as xe
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER,mticker
import cmaps
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from cartopy import feature as cfeature
from matplotlib import patches
import cartopy

#读取数据---------------------------------------------
data=np.loadtxt("C:/Users/Zhao Qun/Desktop/SURF_CLI_CHN_MUL_DAY-TEM-12001-200809.TXT")
xday=data[::30,:]       #选取9月1日所有站点数据
temp=xday[:,7]/10   #温度
    
#print(temp)
#demlores=xr.open_dataset("C:/Users/Zhao Qun/Desktop/DemLoRes.nc")
#demhires=xr.open_dataset("C:/Users/Zhao Qun/Desktop/DemHiRes.nc")
#lat_low,lon_low=demlores.lat,demlores.lon

#print(lat)

hires=xr.open_dataset('C:/Users/Zhao Qun/Desktop/zhaoqun_HiRes.nc')
lat_hi=hires.lat
lon_hi=hires.lon
temp_hi=hires.tem
#print(temp_hi)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

fig=plt.figure(figsize=(50,45),dpi=100)
proj=ccrs.PlateCarree()  #central_longitude=180
ax=fig.add_subplot(111,projection=proj)
ax.contourf(lon_hi, lat_hi, temp_hi-273.15, cmap=cmap1,crs=ccrs.PlateCarree(),levels=np.linspace(-15,35,11))
#ax.set_extent([70, 140, 10, 55],crs=proj)
ax.coastlines(resolution="50m", linewidth=1)              # resolution 分辨率（valid scales）--110m、50m、10m。
ax.add_feature(cfeature.LAND, facecolor='lightgrey')  # 设置陆地颜色为灰色,[0.5, 0.5, 0.5]为灰色
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.tick_params(size=6, labelsize=18)
ax.set_xlabel('longitude',size=18)
ax.set_ylabel('latitude',size=18)
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks(np.arange(70,141,10), crs=proj)
ax.set_yticks(np.arange(10,60,10), crs=proj)

#将站点经纬度转换为每度每分，为60进制
lat=xday[:,1:2]//100+(xday[:,1:2]-xday[:,1:2]//100*100)/60
lon=xday[:,2:3]//100+(xday[:,2:3]-xday[:,2:3]//100*100)/60
lat=lat.ravel()     #转为一维数组
lon=lon.ravel()
#将站点以散点图的形式绘制出来
ax.scatter(lon[:] , lat[:] , marker='*' , s=12 , color = "darkviolet" , zorder = 3)


#绘制colorbar----------------------------------------
ax0=fig.add_axes([0.2,0.08,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-15, vmax=35)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature',  rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-15, 35, 11))
fc1.ax.tick_params(labelsize=15)      #调用colorbar的ax属性

#子图位置和标题-------------------------------------
plt.subplots_adjust(left=0.080,
                    bottom=0.200,
                    right=0.920,
                    top=0.92,
                    wspace=0.1,      #子图间垂直距离
                    hspace=0.15     #子图间水平距离
                   )
plt.suptitle('HiRes', fontsize=20, x=0.5, y=0.95)# x,y=0.5,0.98(默认)

plt.show()

#print(np.min(temp_hi), np.max(temp_hi))
temp_ave=np.mean(temp_hi-273.15)
print(temp_ave)




#print(lat,lon,lat_low,lon_low)
'''print(lat)
data0 = {
    'value': temp,  # 随机生成10个1到100之间的整数
    'lon': lon,  # 随机生成10个-180到180之间的经度值
    'lat': lat  # 随机生成10个-90到90之间的纬度值
    }

# 创建Pandas DataFrame
df = pd.DataFrame(data0)
print(df)

#使用双线性插值方法
grid_values = griddata(
    points=(lon, lat),    # 输入点（站点经纬度）
    values=temp,           # 站点温度
    xi=(lon_low, lat_low), # 插值目标网格点
    method='linear'          # 插值方法
)
'''






'''
temp_da = xr.DataArray(
    temperature,
    dims=['lat', 'lon'],
    coords={'lat': lat, 'lon': lon},
    attrs={'units': '°C', 'description': 'Surface Temperature'}
    )
'''
