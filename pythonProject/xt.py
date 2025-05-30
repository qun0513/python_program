import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
#from wrf import getvar, interpline, CoordPair,get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,geo_bounds

#兰伯特投影
import proplot as plot
import xarray as xr
import netCDF4
t=netCDF4.Dataset('D:/毕业设计/wrfout_d01_2021-07-19_12_00_00')
t2=t.t2
#t2 = getvar(ncfile, "T2")
t2 = t2 - 273.15

lat =t2.XLAT
lon = t2.XLONG

proj = plot.Proj('lcc', lon_0=38)
fig,axs = plot.subplots(ncols=1,width=10,height=10,projection=proj,facecolor='white')

axs.format(reso='hi',coast=True,metalinewidth=2,coastlinewidth=0.8,
           lonlim=(70, 140),latlim=(11, 55),                      #82.5:66.5,15.5:60.5
           title='Barents sea sea ice\ndifference map in winter',
           titlesize=15,titlepad=10,facecolor='white'
           )
axs.tick_params(labelsize=20)
#Barents Sea sea ice difference map in winter
#Barents Sea sea ice difference map in autumn
gl=axs[0].gridlines(
    xlocs=np.arange(-180, 180 + 1, 10), ylocs=np.arange(-90, 90 + 1, 5),
    draw_labels=True, x_inline=False, y_inline=False,
    linewidth=0.5, linestyle='--', color='none')

gl.top_labels = False
gl.right_labels = False
gl.rotate_labels =0
gl.xlabel_style = {'size':13, 'color':'black'}
gl.ylabel_style = {'size':13, 'color':'black'}

#dif=shw-slw
#print(dif)

plt.contourf(lon, lat, t2, levels = np.arange(-0.7,0.1,0.1),cmap=mpl.cm.YlOrBr_r,extend='both')
#MPL_PiYG_r
#plt.title('2019-01-16',position=(0.5,0.9),loc='center',fontsize=20)

#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=axs[0], orientation="vertical", pad=0.0001, aspect=16, shrink=0.8,drawedges=True)
cb.set_label('',size=14,rotation=0,labelpad=5,fontsize=20)
cb.ax.tick_params(labelsize=10)
plt.show()
#plt.savefig("fig/lcc_zq.png", dpi=400, bbox_inches='tight')
