#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
ds=xr.open_dataset('D:/data/AO and Winter air pollution in China/wind-big-200.nc') #200hPa
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

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
with open('D:/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
fig = plt.figure(figsize=[8,8])
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.LAND.with_scale('50m'))
for line in borders:
    ax.plot(line[0::2],line[1::2],'-',color='gray',transform=ccrs.Geodetic())
ax.gridlines(linestyle='--')
ax.set_extent([50,150,5,60])
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linestyle='--')
gl.xlabels_top = False  
gl.ylabels_right = False  
gl.xformatter =LONGITUDE_FORMATTER
gl.yformatter =LATITUDE_FORMATTER 
lat=ds.latitude
lon=ds.longitude
u=u1-u3
v=v1-v3



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

contourf=ax.contourf(lon,lat,u,levels=np.arange(-12.5,12.5,2.5),cmap=new_cmap1,transform=ccrs.PlateCarree())
plt.colorbar(contourf,ax=ax,fraction=0.033,pad=0.02)
x=lon[::5]
y=lat[::5]
x,y=np.meshgrid(x,y)
u= u[::5,::5]
v= v[::5,::5]
u=u.values
v=v.values
plt.streamplot(x,y,u,v,linewidth=1,density=1,color='k',transform=ccrs.PlateCarree())
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




