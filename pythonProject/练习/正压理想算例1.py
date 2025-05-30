import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
#import matplotlib as mpl
import imageio
import os, sys
import glob
from PIL import Image, ImageOps

ds=xr.open_dataset("D:/jisuan/share/swm_rh.360x180/swm_rh.360x180.h0.nc")
print(ds)
#print(ds.u)
u=ds.u[0,:,:]#.mean(dim='time')
print(u)
z=ds.z[:,:,:].mean(dim='time')
#v=ds.v[10,:,:]#.mean(dim='time')
#lat=ds.lat
#lon=ds.lon
lat=np.arange(-90,90,1)
lon=np.arange(0,360,1)
#print(lat)
#print(lon)
lon,lat=np.meshgrid(lon,lat)

lon=ds.u.lon
lat=ds.u.lat

#绘图--------------------------------------------------------------------------
#fig, ax = plt.subplots()
'''
fig, ax = plt.subplots()
import cmaps
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cmap1 = cmaps.BlueDarkRed18  # BlueDarkRed18  GMT_panoply
list_cmap1 = cmap1(np.linspace(0, 1, 256))
cp = ListedColormap(list_cmap1, name='cp')
#norm=mpl.colors.Normalize(vmin=,vmax=100)
'''
video=[]
outputpath='D:/jisuan/算例/正压理想算例2/z.gif'


for i in np.arange(0,100):
    fig, ax = plt.subplots()
    import cmaps
    from matplotlib.colors import ListedColormap

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 SimHei FangSong(可能更普适)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    cmap1 = cmaps.BlueDarkRed18  # BlueDarkRed18  GMT_panoply
    list_cmap1 = cmap1(np.linspace(0, 1, 256))
    cp = ListedColormap(list_cmap1, name='cp')
    # norm=mpl.colors.Normalize(vmin=,vmax=100)
    u = ds.u[i, :, :]  # .mean(dim='time')
    v = ds.v[i, :, :]  # .mean(dim='time')
    z = ds.z[i, :, :]  # .mean(dim='time')
    rr=ax.contourf(lon,lat, z,levels=np.linspace(8000,10900,30),extend='both',cmap=cp,zorder=0)  #MPL_PiYG_r
    plt.title('正压理想算例2_z{}'.format(i),loc='center',fontsize=12,pad=12)  ##ST-Significance test
    cb = plt.colorbar(rr,ticks=[8000,8500,9000,9500,10000,10500],orientation="vertical", pad=0.07, aspect=20, shrink=1)
    cb.set_label('',size=5,rotation=0,labelpad=5,fontsize=15)
    #cb.set_xticks([])--------------
    cb.ax.tick_params(labelsize=12)

    plt.savefig("D:/jisuan/算例/正压理想算例2/z{}.png".format(i))
    video.append(imageio.imread("D:/jisuan/算例/正压理想算例2/z{}.png".format(i)))
    plt.cla()
    print(i)

#plt.show()
imageio.mimsave(outputpath,video,"GIF",duration=0.2)


#---动图方法二-----------------------------------------------------------
'''
def png2gif(filelist, name, duration=0.5):
    frames = []
    for img in filelist:
        #crop_margin(img, img)
        frames.append(imageio.imread(img))
    imageio.mimsave(name,frames,'GIF',duration=0.2)


path =r"D:/jisuan/算例/正压理想算例2"
Toplist = glob.glob(os.path.join(path, "*png"))    #1、11、111；2、21、22、223
Toplist.sort()

png2gif(Toplist, "D:/jisuan/算例/正压理想算例2/v.gif")
'''
#plt.show()