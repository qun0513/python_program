#数据读取---------------------------------------------------------------
import os
import xarray as xr
import numpy as np
import math
import  matplotlib.pyplot as plt
path = 'D:/dyhl/data_60_89_CTL'         #控制实验
#path2 = 'D:/dyhl/data_60_89_Wind'        #风应力实验
path2 = 'D:/dyhl/data_60_89_Water'       #淡水通量实验

#数据读取--------------------------------------------------------------

ff=[]
for filename in os.listdir(path):
    full_path = os.path.join(path,filename)
    f = xr.open_dataset(full_path,decode_times=False)
    ff.append(f.ss[0,:,:,140:180].mean(dim='lon'))     #变量
print(f.ss)
fff=np.array(ff).reshape(360,30,115)
print(f.ss[0,:,:,140:180].data)

ff2=[]
for filename in os.listdir(path2):
    full_path = os.path.join(path2,filename)
    f2 = xr.open_dataset(full_path,decode_times=False)
    ff2.append(f2.ss[0,:,:,140:180].mean(dim='lon'))     #变量
print(f2.ss)
fff2=np.array(ff2).reshape(360,30,115)
print(f.ss[0,:,:,140:180].data)
'''
#写入文件-------------------
with open('D:/dyhl/thc_ts.txt','w') as outfile:
    for slice_2d in fff:
        np.savetxt(outfile, slice_2d) # fmt='%d' ; fmt='%.2f' ; delimiter=',' 分隔符

fff=np.loadtxt('D:/dyhl/thc_ts.txt').reshape(360,30,115)
print(fff)

f = xr.open_dataset("D:\dyhl\data_60_89_CTL\MMEAN0060-01.nc",decode_times=False)
'''

lev=f.ts.lev
lev=lev-5
lat=f.ts.lat

lat,lev=np.meshgrid(lat,lev)

#绘图-------------------------------------------------------------------

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.GMT_haxby
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')


#绘图------------------------------------------------------------------------
'''
fig,ax=plt.subplots()
plt.contourf(lat, lev, fff[0,:,:],levels=np.linspace(-4,32,10),cmap=cp)#,levels=np.linspace(-0.042,0.042,13))
#levels=np.linspace(-0.05,0.05,11)     levels=np.linspace(-0.02,0.02,9)
#MPL_PiYG_r\GMT_polar\BlueDarkRed18\GMT_panoply

plt.title('控制实验纬向平均温度层结',loc='center',fontsize=24,pad=15)
ax.set_xlabel('Latitude(degrees)',fontsize=20,labelpad=10)
ax.set_ylabel('Depth(meters)',fontsize=20,labelpad=10)
ax.tick_params(size=6,labelsize=15)
ax.set_xticks(np.arange(-60,90,15))

ax.set_yscale('symlog')
#ax.set_ylim(-5010,-10)
ax.set_yticks([-5005,-2555,-1005,-505,-305,-205,-105,-55,-25,-15])
ax.set_yticklabels([-5000,-2500,-1000,-500,-300,-200,-100,-50,-20,-10])

cb = plt.colorbar(ax=ax, orientation="vertical",norm='normlize',pad=0.025, aspect=40, shrink=1)
cb.set_label('Temperature',size=3,rotation=90,labelpad=8,fontsize=20)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=15)

cmarker=plt.contour(lat,lev,fff[0,:,:],[4,20],colors='b',lw=3)
plt.clabel(cmarker,fontsize=16,colors=['k','k'],fmt='%.2f')

print("Hello,world!")
plt.show()
'''

'''
#循环绘图
#fig, ax = plt.subplots()
for i in np.arange(0,360,36):
    fig, ax = plt.subplots()
    plt.contourf(lat, lev, fff[i, :, :], levels=np.linspace(-4,32,10), cmap=cp)  # ,levels=np.linspace(-0.042,0.042,13))
    # levels=np.linspace(-0.05,0.05,11)     levels=np.linspace(-0.02,0.02,9)
    # MPL_PiYG_r\GMT_polar\BlueDarkRed18\GMT_panoply

    plt.title('淡水通量实验实验纬向平均温度层结', loc='center', fontsize=24, pad=15)
    

    cb = plt.colorbar(ax=ax, orientation="vertical", norm='normlize', pad=0.025, aspect=40, shrink=1)
    cb.set_label('Temperature', size=3, rotation=90, labelpad=8, fontsize=20)
    # cb.set_xticks([])
    cb.ax.tick_params(labelsize=15)

    cmarker = plt.contour(lat, lev, fff[i, :, :], [4, 20], colors='b')
    plt.clabel(cmarker, fontsize=16, colors=['k', 'k'], fmt='%.2f')

    plt.tight_layout()

    plt.savefig("D:/dyhl/midfig/thc_Tlevel{}.png".format(i))
    plt.cla()
    print(i)
'''
from matplotlib import patches
import matplotlib as mpl
fig=plt.figure()

ax=fig.add_subplot(331)   #之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
ax.contourf(lat, lev, fff2[0, :, :], cmap=cmap1, levels=np.linspace(30, 40, 10)) #ts (-4,36)
# ax.set_title(' ', pad=12)
ax.set_xlabel('Latitude(degrees)', fontsize=20, labelpad=10)
ax.set_ylabel('Depth(meters)', fontsize=20, labelpad=10)
ax.tick_params(size=6, labelsize=15)
ax.set_xticks(np.arange(-60, 90, 15))

ax.set_yscale('symlog')
# ax.set_ylim(-5010,-10)
ax.set_yticks([-5005, -2555, -1005, -505, -205, -105, -55, -25, -15])
ax.set_yticklabels([-5000, -2500, -1000, -500, -200, -100, -50, -20, -10])
ax.set_xlabel(' ')
ax.set_xticks([])
ax.set_ylabel(' ')

ax.annotate(str(1),xy=(-0.05,1.1),xycoords='axes fraction',c='r',fontsize=16,ha='left',va='top')
boxa1 = patches.Rectangle((70, -200), 18, 190, linewidth=2, linestyle='--', zorder=2,
                                                      edgecolor='r', facecolor='none',
                                                       )  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
ax.add_patch(boxa1)

#k=[0,11,35,59,119,179,239,299,359]
k=[12,36,60,120,180,240,300,348]
for i in np.arange(0,8):
    j=332+i

    ax=fig.add_subplot(j)   #之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
    ax.contourf(lat, lev, fff[k[i], :, :], cmap=cmap1, levels=np.linspace(30, 40, 10)) #ts (-4,36)
    # ax.set_title(' ', pad=12)
    ax.set_xlabel('Latitude(degrees)', fontsize=20, labelpad=10)
    ax.set_ylabel('Depth(meters)', fontsize=20, labelpad=10)
    ax.tick_params(size=6, labelsize=15)
    ax.set_xticks(np.arange(-60, 90, 15))

    ax.set_yscale('symlog')
    # ax.set_ylim(-5010,-10)
    ax.set_yticks([-5005, -2555, -1005, -505, -205, -105, -55, -25, -15])
    ax.set_yticklabels([-5000, -2500, -1000, -500, -200, -100, -50, -20, -10])

    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 :
        ax.set_xlabel('')
        ax.set_xticks([])
    if i==0 or i== 1 or i==3 or i==4 or i==6 or i==7 :
        ax.set_ylabel('')
        ax.set_yticks([])
    if i==0 or i==5:
        ax.set_ylabel('')
    if i== 5 or i==7:
        ax.set_xlabel('')

    cmarker = plt.contour(lat, lev, fff[k[i], :, :], [4, 20], colors='b')
    plt.clabel(cmarker, fontsize=16, colors=['k', 'k'], fmt='%.2f')

    m=k[i]+1
    ax.annotate(str(m),xy=(-0.05,1.1),xycoords='axes fraction',c='b',fontsize=16,ha='left',va='top')
    boxa1 = patches.Rectangle((70, -200), 18, 190, linewidth=2, linestyle='--', zorder=2,
                                                      edgecolor='r', facecolor='none',
                                                       )  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
    ax.add_patch(boxa1)
    #风应力实验添加-------------

    #boxa = patches.Rectangle((-76, -1000), 18, 985, linewidth=2, linestyle='--', zorder=2,
    #                         edgecolor='r', facecolor='none',
    #                         )  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
    #ax.add_patch(boxa)
    #boxa1 = patches.Rectangle((-5, -45), 10, 35, linewidth=2, linestyle='--', zorder=2,
    #                        edgecolor='g', facecolor='none',
    #                         )  # , transform=ccrs.PlateCarree() ccrs.LambertConformal()
    #ax.add_patch(boxa1)

#限定子图的位置等参数
plt.subplots_adjust(left=0.080,
                    bottom=0.165,
                    right=0.920,
                    top=0.92,
                    wspace=0.1, #子图间垂直距离
                    hspace=0.15  #子图间水平距离
                   )

plt.suptitle('淡水通量实验大西洋（西经80-0）纬向平均的盐度层结',fontsize=20,x=0.5,y=0.98)# x,y=0.5,0.98(默认)

ax0=fig.add_axes([0.2,0.06,0.6,0.03])
norm =mpl.colors.Normalize(vmin=30, vmax=40)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('Temperature', size=3, rotation=0, labelpad=5, fontsize=20,loc='center')
ax1=fc1.ax                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(30, 40, 10))
fc1.ax.tick_params(labelsize=15)   #调用colorbar的ax属性
plt.show()

'''
#绘制动图
import imageio
video=[]
for i in [0,1,12,24,36,72,108,144,180,216,252,288,324]:
    video.append(imageio.imread("D:/dyhl/midfig/wind_Tlevel{}.png".format(i)))

imageio.mimsave("D:/dyhl/midfig/wind_Tlevel.gif", video, "GIF", duration=1)
'''

