import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

a=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
print(a)
print('XXX')
a=a.sic.loc[:,89.5:40.5,-179.5:179.5]
#loc基于索引名称来获取
#iloc基于索引位置来获取
print(a)
b0=a.isel(time=slice(1308,1309))  #1308,1309  1788,1789
b1=a.isel(time=slice(1788,1789))  #1308,1309  1788,1789
print(b0)
print(b1)
print('XXXXXXXXXXX')
b=b1.data-b0.data
lon=a.longitude
lat=a.latitude
'''
d=b.mean(dim='longitude')
e=d.mean(dim='latitude')
print(e)
fig,ax=plt.subplots()
x=np.linspace(1979,2019,492)
ax.plot(x,e)
plt.show()
'''

import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import proplot as plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math

proj = plot.Proj('npstere', central_longitude=90)

fig, ax = plot.subplots(proj=proj)
ax.format(labels=False, grid=False, coast=True, metalinewidth=1.5, boundinglat=40)

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1, color='lightgrey', x_inline=True, y_inline=True,
                  xlocs=np.arange(-180, 180, 45))

ins = ax.inset_axes([0., 0., 1, 1], proj='polar', zorder=2)
ins.set_theta_zero_location("S", -90)


# 画经纬度label,很难自动生成：
def format_fn(tick_val, tick_pos):
    t = tick_val / math.pi / 2 * 360
    lons = LongitudeFormatter(zero_direction_label=False).__call__(t, tick_pos)
    return lons


formatter = mticker.FuncFormatter(format_fn)
ins.format(grid=False, rgrid=False, rborder=True, rlocator=[0], facecolor='white', alpha=0, thetadir=1,
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
ins.tick_params(labelsize=12)  #刻度值大小-----------------------------

mtick_gen(ins, 6, 45, color="k", lw=1, direction='out')
mtick_gen(ins, 4, 15, color="k", lw=1, direction='out')

import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueDarkRed18  #BlueDarkRed18  GMT_panoply
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

#fig,ax= plt.subplots(figsize=(50,45),dpi=100)
#ax = plt.axes(projection=ccrs.NorthPolarStereo())
#ax.set_extent([-179.5,179.5,89.5,60.5])
#ax.coastlines(resolution="50m",linewidth=1)  #resolution:110m\ 50m\ 10m

b1=np.array(b1).reshape(50,360)
plt.contourf(lon, lat, b1,levels=np.arange(0,1.4,0.1),cmap=cmaps.WhiteYellowOrangeRed)
#MPL_PiYG_r
plt.title('2019-01-16',position=(0.5,0.9),loc='center',fontsize=20)

#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.0001, aspect=16, shrink=0.8)
cb.set_label('',size=14,rotation=0,labelpad=5,fontsize=20)
cb.ax.tick_params(labelsize=10)

plt.savefig('D:/GC/SIC_20190116.jpg')

