import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib import patches
#fig,ax=plt.subplots(3,3)
#ax[0][0].xaxis.set_tick_params(labelsize=12)  #用错误来试出正确 fontsize
#ax[0][0].plot(np.arange(0,5),np.arange(0,10,2))
#ax[1][1].yaxis.tick_right()
#ax[1][1].plot(np.arange(0,5),np.arange(0,15,3))

import proplot as plot
proj = plot.Proj('lcc', lon_0=38)
fig,axs = plot.subplots(ncols=1,width=10,height=10,projection=proj,facecolor='white')

axs.format(reso='hi',coast=True,metalinewidth=2,coastlinewidth=0.8,
           lonlim=(0,180),latlim=(-50,83),                      #82.5:66.5,15.5:60.5
           title='Barents sea sea ice\ndifference map in winter',
           titlesize=15,titlepad=10,facecolor='white',zorder=1
           )

boxa = patches.Rectangle((110,20),20,10, linewidth=2, linestyle='-', zorder=2,
                         edgecolor='b', facecolor='none', transform=ccrs.PlateCarree())#, transform=ccrs.PlateCarree() ccrs.LambertConformal()
axs.add_patch(boxa)

plt.show()
#['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor',
# 'zorder', 'gridOn', 'tick1On', 'tick2On', 'label1On', 'label2On',
# 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft',
# 'labelbottom', 'labelright', 'labeltop', 'labelrotation', 'grid_agg_filter',
# 'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box',
# 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_contains',
# 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 'grid_data',
# 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gid', 'grid_in_layout',
# 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker',
# 'grid_markeredgecolor', 'grid_markeredgewidth', 'grid_markerfacecolor',
# 'grid_markerfacecoloralt', 'grid_markersize', 'grid_markevery', 'grid_path_effects',
# 'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params',
# 'grid_snap', 'grid_solid_capstyle', 'grid_solid_joinstyle', 'grid_transform',
# 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 'grid_zorder', 'grid_aa',
# 'grid_c', 'grid_ds', 'grid_ls', 'grid_lw', 'grid_mec', 'grid_mew', 'grid_mfc',
# 'grid_mfcalt', 'grid_ms']
