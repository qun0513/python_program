import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
ds=xr.open_dataset('D:/XC/wind-big-200.nc')
lat=ds.latitude
lon=ds.longitude
#筛选出1979.12--2019.12
data3=ds.isel(time=slice(2,123))
#40年风场气候态平均w3
#筛选出1980.1--2019.12
data30=ds.isel(time=slice(3,123))
w3=data30.mean(dim='time')
u3=w3.u
v3=w3.v
##高值月份
d10=data3.isel(time=40).u
d10=d10+data3.isel(time=28).u
d10=d10+data3.isel(time=29).u
d10=d10+data3.isel(time=32).u
d10=d10+data3.isel(time=81).u
d10=d10+data3.isel(time=82).u
d10=d10+data3.isel(time=96).u
d10=d10+data3.isel(time=108).u
d10=d10+data3.isel(time=53).u
d10=d10+data3.isel(time=111).u
d10=d10+data3.isel(time=27).u
d10=d10+data3.isel(time=39).u
d10=d10+data3.isel(time=67).u
d10=d10+data3.isel(time=102).u
d10=d10+data3.isel(time=10).u
d10=d10+data3.isel(time=36).u
d10=d10+data3.isel(time=57).u
d10=d10+data3.isel(time=9).u

d1=data3.isel(time=40).v
d1=d1+data3.isel(time=28).v
d1=d1+data3.isel(time=29).v
d1=d1+data3.isel(time=32).v
d1=d1+data3.isel(time=81).v
d1=d1+data3.isel(time=82).v
d1=d1+data3.isel(time=96).v
d1=d1+data3.isel(time=108).v
d1=d1+data3.isel(time=53).v
d1=d1+data3.isel(time=111).v
d1=d1+data3.isel(time=27).v
d1=d1+data3.isel(time=39).v
d1=d1+data3.isel(time=67).v
d1=d1+data3.isel(time=102).v
d1=d1+data3.isel(time=10).v
d1=d1+data3.isel(time=36).v
d1=d1+data3.isel(time=57).v
d1=d1+data3.isel(time=9).v
u1=d10/18
v1=d1/18
#共有18个高值，19个低值
##低值月份
d20=data3.isel(time=23).u
d20=d20+data3.isel(time=26).u
d20=d20+data3.isel(time=73).u
d20=d20+data3.isel(time=48).u
d20=d20+data3.isel(time=94).u
d20=d20+data3.isel(time=63).u
d20=d20+data3.isel(time=99).u
d20=d20+data3.isel(time=77).u
d20=d20+data3.isel(time=11).u
d20=d20+data3.isel(time=55).u
d20=d20+data3.isel(time=78).u
d20=d20+data3.isel(time=1).u
d20=d20+data3.isel(time=17).u
d20=d20+data3.isel(time=16).u
d20=d20+data3.isel(time=93).u
d20=d20+data3.isel(time=91).u
d20=d20+data3.isel(time=90).u
d20=d20+data3.isel(time=20).u
d20=d20+data3.isel(time=92).u

d2=data3.isel(time=23).v
d2=d2+data3.isel(time=26).v
d2=d2+data3.isel(time=73).v
d2=d2+data3.isel(time=48).v
d2=d2+data3.isel(time=94).v
d2=d2+data3.isel(time=63).v
d2=d2+data3.isel(time=99).v
d2=d2+data3.isel(time=77).v
d2=d2+data3.isel(time=11).v
d2=d2+data3.isel(time=55).v
d2=d2+data3.isel(time=78).v
d2=d2+data3.isel(time=1).v
d2=d2+data3.isel(time=17).v
d2=d2+data3.isel(time=16).v
d2=d2+data3.isel(time=93).v
d2=d2+data3.isel(time=91).v
d2=d2+data3.isel(time=90).v
d2=d2+data3.isel(time=20).v
d2=d2+data3.isel(time=92).v

u2=d20/19
v2=d2/19



u=u1-u3
v=v1-v3
uu=u2-u3
vv=v2-v3


#colormap


import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

cmap1=mpl.cm.YlOrBr_r
cmap2=mpl.cm.Blues
list_cmap1=cmap1(np.linspace(0,1,6))
list_cmap2=cmap2(np.linspace(0,1,6))

cmap7=mpl.cm.bwr_r
list_cmap7=cmap7(np.linspace(0,1,12))
list_cmap7=ListedColormap(list_cmap7[6:12],name='list_cmap7')
list_cmap7=list_cmap7(np.linspace(0,1,6))

new_color_list1=np.vstack((list_cmap1,list_cmap2))
new_cmap1=ListedColormap(new_color_list1,name='new_cmap2')  #colormap1

new_color_list2=np.vstack((list_cmap1,list_cmap7))
new_cmap2=ListedColormap(new_color_list2,name='new_cmap2')  #colormap2


fig,axes= plt.subplots(nrows=1,ncols=2,figsize=(50,45),dpi=100)
axes[0] = plt.axes(projection=ccrs.PlateCarree())
axes[0].set_extent([-30,150,0,90])
axes[0].coastlines(resolution="50m",linewidth=1)
axes[1] = plt.axes(projection=ccrs.PlateCarree())
axes[1].set_extent([-30,150,0,90])
axes[1].coastlines(resolution="50m",linewidth=1)

# Add gridlines
gl0 = axes[0].gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linewidth=1,
                  color='black',
                  linestyle='--'
                  )
gl0.xlabels = False
gl0.ylabels_right = False
gl0.xlines = False
gl0.ylines=False




#gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
#gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl0.xformatter = LONGITUDE_FORMATTER
gl0.yformatter = LATITUDE_FORMATTER
gl0.xlabel_style = {'size':10, 'color':'black'}
gl0.ylabel_style = {'size':10, 'color':'black'}

gl1 = axes[1].gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linewidth=1,
                  color='black',
                  linestyle='--'
                  )
gl1.xlabels = False
gl1.ylabels_right = False
gl1.xlines = False
gl1.ylines=False




#gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
#gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl1.xformatter = LONGITUDE_FORMATTER
gl1.yformatter = LATITUDE_FORMATTER
gl1.xlabel_style = {'size':10, 'color':'black'}
gl1.ylabel_style = {'size':10, 'color':'black'}

#绘制中国地图
from cartopy.io.shapereader import Reader
proj=ccrs.PlateCarree()
axes[0]=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
axes[0].add_feature(china,lw=0.5,zorder=2)
grid_lines0=axes[0].gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
axes[0].add_feature(cfeature.LAND.with_scale('50m'))
axes[0].add_feature(cfeature.OCEAN.with_scale('50m'))

axes[1]=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
axes[1].add_feature(china,lw=0.5,zorder=2)
grid_lines1=axes[1].gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
axes[1].add_feature(cfeature.LAND.with_scale('50m'))
axes[1].add_feature(cfeature.OCEAN.with_scale('50m'))


clevs = np.linspace(-6,6,13)  #始、终都包含。 arange只包含始点
axes[0].contourf(lon, lat, u, clevs, transform=ccrs.PlateCarree(),cmap=new_cmap2,extend='both')

axes[1].contourf(lon, lat, uu, clevs, transform=ccrs.PlateCarree(),cmap=new_cmap2,extend='both')
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)MPL_PiYG_r
'''
ax.clabel(m,inline=True,colors='r',
          manual=[(120,15),(120,30),(120,45),(120,60)],
          fontsize=20)
'''
#由于cmap的颜色映射表是有固定存储顺序的数组，所以我们可以在需要的时候翻转cmap与数值的对应顺序，翻转命令为在颜色映射表的字符串最后加上’_r’
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=axes[0], orientation="vertical", pad=0.04, aspect=16,shrink=0.8)
cb.set_label('m/s',size=14,rotation=0,labelpad=15,fontsize=20)
cb.ax.tick_params(labelsize=10)

cb1 = plt.colorbar(ax=axes[1], orientation="vertical", pad=0.04, aspect=16,shrink=0.8)
cb1.set_label('m/s',size=14,rotation=0,labelpad=15,fontsize=20)
cb1.ax.tick_params(labelsize=10)

uwnd = u.sortby("latitude",ascending=True)
vwnd = v.sortby("longitude",ascending=True)
winddata_dense_raw = xr.Dataset(data_vars={"uwnd": uwnd, "vwnd": vwnd})
winddata_dense = winddata_dense_raw.assign(windspeed = np.hypot(winddata_dense_raw.uwnd, winddata_dense_raw.vwnd))
winddata_thin = winddata_dense.thin(latitude=2, longitude=2)
winddata_thin.plot.streamplot(x="longitude", y="latitude", u="uwnd", v="vwnd",
                              #hue="windspeed",
                                            ax=axes[0],

                                            #cbar_kwargs = {
                                            #  "aspect": 40,
                                            #   "pad": 0.15,
                                            #  "label": "Wind Speed [m/s]",
                                            # },
                                            linewidth = 1.3,
                                            #cmap = "plasma",
                                            density=1.5
                                            )

uwnd = uu.sortby("latitude",ascending=True)
vwnd = vv.sortby("longitude",ascending=True)
winddata_dense_raw = xr.Dataset(data_vars={"uwnd": uwnd, "vwnd": vwnd})
winddata_dense = winddata_dense_raw.assign(windspeed = np.hypot(winddata_dense_raw.uwnd, winddata_dense_raw.vwnd))
winddata_thin = winddata_dense.thin(latitude=2, longitude=2)
winddata_thin.plot.streamplot(x="longitude", y="latitude", u="uwnd", v="vwnd",
                              #hue="windspeed",
                                            ax=axes[1],

                                            #cbar_kwargs = {
                                            #  "aspect": 40,
                                            #   "pad": 0.15,
                                            #  "label": "Wind Speed [m/s]",
                                            # },
                                            linewidth = 1.3,
                                            #cmap = "plasma",
                                            density=1.5
                                            )

plt.rcParams['font.sans-serif']=['FangSong'] #用来正常显示中文标签 SimHei
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

axes[0].title('a.1980-2020冬季高值年风速距平等值线图',size=20,pad=20)
axes[1].title('a.1980-2020冬季低值年风速距平等值线图',size=20,pad=20)
plt.show()
