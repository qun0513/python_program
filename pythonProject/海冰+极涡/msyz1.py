import xarray as xr
import numpy as np
import pandas as pd
import matplotlib             #解决Linux无法可视化的问题
matplotlib.use('Agg')         #
import matplotlib.pyplot as plt

ua=xr.open_dataset('/home2/zhangcy/zhangcy3/xum/data/pamip/ua_sondjfma_pamip_AWI-CM-1-1-MR.nc')
print(ua)
ua1=ua.ua1.mean(dim='ensemble')
u1=ua1.mean(dim='lon')
u11=u1.sel(time=u1.time.dt.month.isin([12]))
u111=u11.sel(time=u11.time.dt.year.isin([2000]))
u12=u1.sel(time=u1.time.dt.month.isin([1,2]))
u122=u12.sel(time=u12.time.dt.year.isin([2001]))
uu=
u1=uu.loc[[70000,50000,40000,30000,20000,15000,10000,7000,5000,3000,2000,1000],0:90].mean(dim='mon')

ua2=ua.ua2.mean(dim='ensemble')
u2=ua2.mean(dim='lon')
u2=u2.isel[3:6,:,:].mean(dim='mon')
u2=u2.loc[[70000,50000,40000,30000,20000,15000,10000,7000,5000,3000,2000,1000],0:90].mean(dim='mon')

ua3=ua.ua3.mean(dim='ensemble')
u3=ua3.mean(dim='lon')
u3=u3.isel[3:6,:,:].mean(dim='mon')
u3=u3.loc[[70000,50000,40000,30000,20000,15000,10000,7000,5000,3000,2000,1000],0:90].mean(dim='mon')

ubs=u2-u1
uos=u3-u1


import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueWhiteOrangeRed#BlueDarkRed18   temp_19lev  BlRe（师兄）
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

fig,ax=plt.subplots()
lat=u1.lat;level=u1.plev  #(回改)  levels=np.linspace(-3,3,15),
plt.contourf(lat,level,ubs,cmap=cp)#,levels=np.linspace(-2.6e-5,2.6e-5,11)
#,levels=np.linspace(-3.2e-5,3.2e-5,11)
#,levels=np.linspace(-1.2e-4,1.2e-4,9)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.05, aspect=25, shrink=1)
cb.set_label('u',size=3,rotation=90,labelpad=5,fontsize=15)
#cb.set_ticks([-1.2e-4,-0.9e-4,-0.6e-4,-0.3e-4,0,0.3e-4,0.6e-4,0.9e-4,1.2e-4])
#[-3.2e-5,-2.56e-5,-1.92e-5,-1.28e-5,-0.64e-5,0,0.64e-5,1.28e-5,1.92e-5,2.56e-5,3.2e-5]

print('hello,world!')
plt.savefig('/home/dell/ZQ19/msyz/U_winter.jpg')