import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
a=xr.open_dataset('D:/GC/HadISST_sst_187001-201903.nc')
print(a)
print(a.sst)
sst=a.sst
sst_m=sst.mean(dim='time')
print(sst_m)
lon=sst.longitude
lat=sst.latitude

from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False
#整个过程应该是数组和色表的对应过程
cmap1=mpl.cm.YlOrBr
cmap2=mpl.cm.Blues_r
list_cmap1=cmap1(np.linspace(0,1,10))
list_cmap2=cmap2(np.linspace(0,1,10))

cmap7=mpl.cm.bwr_r
list_cmap7=cmap7(np.linspace(0,1,12))
list_cmap7=ListedColormap(list_cmap7[6:12],name='list_cmap7') #先把它变成一个色表
list_cmap7=list_cmap7(np.linspace(0,1,6))                     #再把它划分成对应的数组

new_color_list1=np.vstack((list_cmap2,list_cmap1))            #对数组进行合并
new_cmap1=ListedColormap(new_color_list1,name='new_cmap2')  #colormap1
                                                            #将数组转化成色表

new_color_list2=np.vstack((list_cmap1,list_cmap7))
new_cmap2=ListedColormap(new_color_list2,name='new_cmap2')  #colormap2

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
fig,ax=plt.subplots(figsize=(50,45))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90])
ax.coastlines(resolution="50m",linewidth=1)  #resolution:110m\50m\10m

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter =LONGITUDE_FORMATTER
gl.yformatter =LATITUDE_FORMATTER
gl.xlabel_style={'size':20}
gl.ylabel_style={'size':20}
clevs = np.arange(-33,34,3)  #np.linspace
plt.contourf(lon, lat, sst_m, clevs, transform=ccrs.PlateCarree(),cmap=mpl.cm.coolwarm,extend='both')
#MPL_PiYG_r
#cmap分类：连续类、分色类、循环类、定性类、混杂类。
cb = plt.colorbar(ax=ax, orientation="horizontal", pad=0.06, aspect=24, shrink=0.8)
cb.set_label('℃',size=14,rotation=0,labelpad=15,fontsize=20)
cb.ax.tick_params(labelsize=20)
ax.set_title('1870-2019年全球平均海平面温度图',fontsize=30,pad=20)
plt.show()



'''
#绘制中国地图
from cartopy.io.shapereader import Reader
proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
ax.add_feature(china,lw=0.5,zorder=2)
ax.add_feature(cfeature.LAND.with_scale('50m'))  #scale:110m\50m\10m
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.tick_params(labelsize=16)
'''

'''
#绘制南海子图
ax_nh=fig.add_axes([0.61,0.128,0.32,0.25],projection=proj)
ax_nh.add_feature(china,lw=0.5,zorder=2)
ax_nh.add_feature(cfeature.LAND.with_scale('50m'))
ax_nh.add_feature(cfeature.OCEAN.with_scale('50m'))
ax_nh.set_extent([105,125,0,25])
'''
