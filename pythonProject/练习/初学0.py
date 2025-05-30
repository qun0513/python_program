#colormap+colorbar

import matplotlib as mpl
import matplotlib.pyplot  as plt
import numpy as np
#fig=plt.figure(figsize=(15,12),dpi=100)
#ax1=fig.add_axes([0,0,0.2,0.2],facecolor='b')
fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(15,12),dpi=100)
x=np.array([1,2,3,4])
y=np.array([1,2,3,4])
z=np.array([1,8,27,64])
w=[1,2,3,4]
axes[0][0].plot(x,y)
axes[1][1].scatter(x,y,s=10,c='tomato')
axes[2][2].bar(x,z)
X,Y=np.meshgrid(x,y)
#plt.contour(X,Y,z,cmaps='jet')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
axes[0][0].set_title('中文')
plt.show()

'''
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

a=xr.open_dataset('D:/XC/wind-big-500.nc')
lon=a.longitude
lat=a.latitude
lon,lat=np.meshgrid(lon,lat)
u=a.u.mean(dim='time')
v=a.v.mean(dim='time')
u=u[::10,::10]
v=v[::10,::10]
ws=np.sqrt(u**2+v**2)
fig=plt.figure()
#ax=fig.subplots(1,2)
# colormap=mpl.cm.Reds
#ax.contourf(cmap=colormap)
ax1=fig.add_axes([0.1,0.2,0.2,0.3],facecolor='r')
ax2=fig.add_axes([0.4,0.2,0.2,0.3],facecolor='green')
ax3=fig.add_axes([0.7,0.2,0.2,0.3],facecolor='m')
ax4=fig.add_axes([0.5,0.6,0.2,0.3])
x=[1,2,3,4]
y=[2,4,6,8]
ax1.set_ylim(0,20)
ax1.set_yticks(np.arange(0,101,10))
ax3.plot(x,y,marker='*')
ax3.grid(which='major',linestyle=':')
ax3.set_xlim(0,5)
ax3.set_xticks(np.arange(0,5,1))
ax3.set_ylim(0,10)
ax3.set_yticks(np.arange(0,11,2))
#fig,axes=plt.subplots(nrows,ncols,figsize,dpi)
z=[1,3,5,7]
ax4.plot(x,z)
plt.show()
'''

