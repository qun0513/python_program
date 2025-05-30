import xarray as xr
import numpy as np
import pandas as pd
import matplotlib             #解决Linux无法可视化的问题
matplotlib.use('Agg')         #
import matplotlib.pyplot as plt

ep=xr.open_dataset('/home2/zhangcy/zhangcy3/xum/data/pamip/ep_sondjfma_pamip_AWI-CM-1-1-MR.nc')
va=xr.open_dataset('/home2/zhangcy/zhangcy3/xum/data/pamip/va_sondjfma_pamip_AWI-CM-1-1-MR.nc')
ta=xr.open_dataset('/home2/zhangcy/zhangcy3/xum/data/pamip/ta_sondjfma_pamip_AWI-CM-1-1-MR.nc')
zg=xr.open_dataset('/home2/zhangcy/zhangcy3/xum/data/pamip/zg_sondjfma_pamip_AWI-CM-1-1-MR.nc')
ua=xr.open_dataset('/home2/zhangcy/zhangcy3/xum/data/pamip/ua_sondjfma_pamip_AWI-CM-1-1-MR.nc')

print(ep.ep1)
# 巴伦支海 ep2 \\ 鄂霍茨克海 ep3 ————————————————————————-——————
ep1=ep.ep1.mean(dim='time')
ep2=ep.ep2.mean(dim='time')
ep3=ep.ep3.mean(dim='time')
    #冬季 3:6
epy11=ep1.loc[2,1,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl3')
epz11=ep1.loc[3,1,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl3')
div11=ep1.loc[4,1,3:6,:,0:90].mean(dim='ncl3')
epy12=ep1.loc[2,2,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl3')
epz12=ep1.loc[3,2,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl3')
div12=ep1.loc[4,2,3:6,:,0:90].mean(dim='ncl3')
epy1=epy11+epy12
epz1=epz11+epz12
div1=div11+div12

epy21=ep2.loc[2,1,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl6')
epz21=ep2.loc[3,1,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl6')
div21=ep2.loc[4,1,3:6,:,0:90].mean(dim='ncl6')
epy22=ep2.loc[2,2,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl6')
epz22=ep2.loc[3,2,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl6')
div22=ep2.loc[4,2,3:6,:,0:90].mean(dim='ncl6')
epy2=epy21+epy22
epz2=epz21+epz22
div2=div21+div22

epy31=ep3.loc[2,1,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl7')
epz31=ep3.loc[3,1,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl7')
div31=ep3.loc[4,1,3:6,:,0:90].mean(dim='ncl7')
epy32=ep3.loc[2,2,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl7')
epz32=ep3.loc[3,2,3:6,[700,500,400,300,200,150,100,70,50,30,20,10],0:90].mean(dim='ncl7')
div32=ep3.loc[4,2,3:6,:,0:90].mean(dim='ncl7')
epy3=epy31+epy32
epz3=epz31+epz32
div3=div31+div32

epy=epy31-epy11
epz=epz31-epz11
div=div31-div11
epz=100*epz
#print(epy)

def enhance(epy,epz,div):
  for i in np.arange(0,12):
    for j in np.arange(0,32):
        if i==7:
            epy[i, j] = epy[i, j] *6
            epz[i, j] = epz[i, j] *6  #0,1,2波*2
            div[i, j] = div[i, j] * 6
        if i>=8:
            epy[i,j]=epy[i,j]*6
            epz[i,j]=epz[i,j]*6      #0,1,2波*3
            div[i, j] = div[i, j] * 6

enhance(epy,epz,div)

import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueWhiteOrangeRed#BlueDarkRed18   temp_19lev  BlRe（师兄）
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

fig,ax=plt.subplots()

lat=div.lat;level=div.plev  #(回改)
print(lat.size,level.size)

#print(lat,level)     #,levels=np.linspace(-9,9,10)
plt.contourf(lat,level,div,levels=np.linspace(-3,3,15),cmap=cp)#,levels=np.linspace(-2.6e-5,2.6e-5,11)
#,levels=np.linspace(-3.2e-5,3.2e-5,11)
#,levels=np.linspace(-1.2e-4,1.2e-4,9)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.05, aspect=25, shrink=1)
cb.set_label('divergence',size=3,rotation=90,labelpad=5,fontsize=15)
#cb.set_ticks([-1.2e-4,-0.9e-4,-0.6e-4,-0.3e-4,0,0.3e-4,0.6e-4,0.9e-4,1.2e-4])
#[-3.2e-5,-2.56e-5,-1.92e-5,-1.28e-5,-0.64e-5,0,0.64e-5,1.28e-5,1.92e-5,2.56e-5,3.2e-5]
cb.ax.tick_params(size=6,labelsize=13)

lat=epy.lat;level=epy.plev                          #150000000(0,1,2波)
plt.quiver(lat,level,epy,epz, width=0.0025,scale=65000000, color='k')
#bha  bla  oha  ola  bhw  blw  ohw  olw              750000000(1)/300000000(0+hl)
ax.invert_yaxis()
ax.set_xlabel('latitude',fontsize=15)
ax.set_ylabel('level',fontsize=15)
ax.set_yscale('symlog')
ax.set_ylim(700,10)

ax.set_yticks([7e2,5e2,3e2,2e2,1e2,5e1,3e1,2e1,1e1,0.8e1])
ax.set_yticklabels([700,500,300,200,100,50,30,20,10,8])
ax.set_title('EP-Flux_winter1',fontsize=18)
ax.tick_params(size=6,labelsize=13)

print('hello,world!')
plt.savefig('/home/dell/ZQ19/msyz/EP-Flux_winter1.jpg')