import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
a=xr.open_dataset('D:/GC/HadISST_sst_187001-201903.nc')
print(a)
b=a.mean(dim='time')
c=b.sst.loc[89.5:60.5,-179.5:179.5]
lon=c.longitude
lat=c.latitude
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import proplot as plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math

proj = plot.Proj('npstere', central_longitude=110)

fig, ax = plot.subplots(proj=proj)
ax.format(labels=False, grid=False, coast=True, metalinewidth=1.5, boundinglat=60)

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1, color='lightgrey', x_inline=True, y_inline=True,
                  xlocs=np.arange(-180, 180, 45))

ins = ax.inset_axes([0., 0., 1, 1], proj='polar', zorder=2)
ins.set_theta_zero_location("S", -110)


# 画经纬度label,很难自动生成：
def format_fn(tick_val, tick_pos):
    t = tick_val / math.pi / 2 * 360
    lons = LongitudeFormatter(zero_direction_label=False).__call__(t, tick_pos)
    return lons


formatter = mticker.FuncFormatter(format_fn)
ins.format(grid=False, rgrid=False, rborder=True, rlocator=[0], facecolor=None, alpha=0, thetadir=1,
           thetalim=(0, 360), gridlabelpad=10
           # ,thetalocator=np.arange(0,360*math.pi/2,45*math.pi/2)#mticker.FixedLocator(np.arange(0,360,45))
           , thetaformatter=formatter  # LongitudeFormatter(zero_direction_label=False)
           , labelsize=7  # ,thetalabels=np.arange(0,360,45))
           )


def mtick_gen(polar_axes, tick_depth, tick_degree_interval=45, color='k', direction='out', **kwargs):
    lim = polar_axes.get_ylim()
    radius = polar_axes.get_rmax()
    if direction == 'out':
        length = tick_depth / 100 * radius
    elif direction == 'in':
        length = -tick_depth / 100 * radius
    for angle in np.deg2rad(np.arange(0, 360, tick_degree_interval)):
        polar_axes.plot(
            [angle, angle],
            [radius + length, radius],
            zorder=500,
            color=color,
            clip_on=False, **kwargs,
        )
    polar_axes.set_ylim(lim)


mtick_gen(ins, 6, 45, color="k", lw=1, direction='out')
mtick_gen(ins, 4, 15, color="k", lw=1, direction='out')



#fig,ax= plt.subplots(figsize=(50,45),dpi=100)
#ax = plt.axes(projection=ccrs.NorthPolarStereo())
#ax.set_extent([-179.5,179.5,89.5,60.5])
#ax.coastlines(resolution="50m",linewidth=1)  #resolution:110m\ 50m\ 10m

#温度单位为0.1
c=c/10

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


plt.contourf(lon, lat, c,levels=np.arange(-80,41,20),cmap=mpl.cm.coolwarm)
#MPL_PiYG_r\YlOrBr_r
plt.title('1870-2019年北极平均海平面温度图',position=(0.5,0.9),loc='center',fontsize=20)

#cmap分类：连续类、分色类、循环类、定性类、混杂类。
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.0001, aspect=16, shrink=0.8)
cb.set_label('',size=14,rotation=0,labelpad=5,fontsize=20)
cb.ax.tick_params(labelsize=10)

plt.show()