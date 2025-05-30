import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmaps
import matplotlib as mpl #cmap=mpl.cm.jet
import matplotlib.ticker as mticker
#a=xr.open_dataset('D:/XC/wind-big-500.nc')
a=xr.open_dataset('D:/XC/wind-big-200.nc')


print(a)
lons=a.longitude
lats=a.latitude
lons=lons[::]
lats=lats[::]
print(lons)
print(lats)
lon,lat=np.meshgrid(lons,lats)
u=a.u.mean(dim='time')
v=a.v.mean(dim='time')
print(u)
print(v)
u=u[::,::]
v=v[::,::]
#print(u)
#print(v)
ws=np.sqrt(u**2+v**2)
ws_d=np.arctan2(u,v)


'''
fig = plt.figure(figsize=(4000,2500),dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-30,150,0,90])
ax.coastlines(resolution="50m",linewidth=1)
# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.ylines=False
#gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
#gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':10, 'color':'black'}
gl.ylabel_style = {'size':10, 'color':'black'}
# Plot windspeed
clevs = np.arange(-30,41,1)
plt.contourf(lon, lat, ws[:,:], clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
#cb.set_label('m/s',size=14,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=10)
'''
#ds1 = xr.open_dataset("D:/XC/wind-big-500.nc", drop_variables = ["time_bnds"]).sel(level = 850).rename({"air": "Tair"})
#ds = ds1.sortby("lat", ascending= True)

#uwnd_raw = xr.open_dataset("..\\uwnd.2020.nc", drop_variables = ["time_bnds"]).sel(level = 850, lon = slice(0, 160), lat=slice(85, 12)).uwnd.isel(time = 5)
#vwnd_raw = xr.open_dataset("..\\vwnd.2020.nc", drop_variables = ["time_bnds"]).sel(level = 850, lon = slice(0, 160), lat=slice(85, 12)).vwnd.isel(time = 5)
'''
uwnd = u.sortby("lat", ascending= True)
vwnd = v.sortby("lat", ascending= True)
winddata_dense_raw = xr.Dataset(data_vars={"uwnd": uwnd, "vwnd": vwnd})
winddata_dense = winddata_dense_raw.assign(windspeed = np.hypot(winddata_dense_raw.uwnd, winddata_dense_raw.vwnd))
winddata_thin = winddata_dense.thin(lat=2, lon=2)
#with plt.xkcd():
'''

fig = plt.figure(figsize=(50,45),dpi=100)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-30,150,0,90])
ax.coastlines(resolution="50m",linewidth=1)

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linewidth=1,
                  color='black',
                  linestyle='--'
                  )
gl.xlabels = False
gl.ylabels_right = False
gl.xlines = False
gl.ylines=False




#gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
#gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':10, 'color':'black'}
gl.ylabel_style = {'size':10, 'color':'black'}
# Plot windspeed

from matplotlib.colors import ListedColormap
import cmaps
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

clevs = np.arange(-6,6,1)
m=plt.contourf(lon, lat, u, clevs, transform=ccrs.PlateCarree(),cmap=new_cmap2,extend='both')
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)MPL_PiYG_r

ax.clabel(m,inline=True,colors='r',
          manual=[(120,15),(120,30),(120,45),(120,60)],
          fontsize=20)
#由于cmap的颜色映射表是有固定存储顺序的数组，所以我们可以在需要的时候翻转cmap与数值的对应顺序，翻转命令为在颜色映射表的字符串最后加上’_r’
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('m/s',size=14,rotation=0,labelpad=15,fontsize=20)
cb.ax.tick_params(labelsize=10)
#lat.values.ravel()
#lon.values.ravel()
#u.values.ravel()
#v.values.ravel()
#uwnd_raw = xr.open_dataset("..\\uwnd.2020.nc", drop_variables = ["time_bnds"]).sel(level = 850, lon = slice(0, 160), lat=slice(85, 12)).uwnd.isel(time = 5)
#vwnd_raw = xr.open_dataset("..\\vwnd.2020.nc", drop_variables = ["time_bnds"]).sel(level = 850, lon = slice(0, 160), lat=slice(85, 12)).vwnd.isel(time = 5)

#绘制流场图

uwnd = u.sortby("latitude",ascending=True)
vwnd = v.sortby("longitude",ascending=True)
winddata_dense_raw = xr.Dataset(data_vars={"uwnd": uwnd, "vwnd": vwnd})
winddata_dense = winddata_dense_raw.assign(windspeed = np.hypot(winddata_dense_raw.uwnd, winddata_dense_raw.vwnd))
winddata_thin = winddata_dense.thin(latitude=2, longitude=2)
winddata_thin.plot.streamplot(x="longitude", y="latitude", u="uwnd", v="vwnd",
                              #hue="windspeed",
                                            ax=ax,

                                            #cbar_kwargs = {
                                            #  "aspect": 40,
                                            #   "pad": 0.15,
                                            #  "label": "Wind Speed [m/s]",
                                            # },
                                            linewidth = 1.3,
                                            #cmap = "plasma",

                                            )

plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签 SimHei
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.title('流场图',size=20)
plt.show()