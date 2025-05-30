
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import cmaps
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from eofs.standard import Eof
from PyEMD import EEMD
from scipy.signal import hilbert, periodogram
#from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import pearsonr
import heapq
import math
from scipy.stats import ttest_1samp

files=f'F:/data/piControl/BCC-CSM2-MR/tos*.nc'
combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
#插值------------------
target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                        ignore_degenerate=True,  # 忽略退化的网格点#???某些网格点可能因为几何形状或数值问题导致插值算法无法正常处理
                        )
regridderdata1= regridder(combined_ds)
# (-180,180) 转为(0,360)------------------
regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
lat_1d = np.array(regridderdata1['lat'].isel(x=0).squeeze().data)  # 提取 NumPy 数组
lon_1d = np.array(regridderdata1['lon'].isel(y=0).squeeze().data)  # 提取 NumPy 数组
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
ds = regridderdata1.sortby('lon')


variable_p=ds.tos[:,70:160,120:280]   # sst pacific_related: 20S-70N, 120E-80W; PDO:20N-70N, 120E-100W
lon=variable_p.lon     #?-----
lat=variable_p.lat

model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
variables=['tos','ua','va','thetao','thetao','thetao']
months=[0,2,5,8,11]
for i in np.arange(0,6):
    mdl=model[i]
    for j in np.arange(0,6):
        variable=variables[j]
        if j==3:
            variable='thetao-50'
        if j==4:
            variable='thetao-100'
        if j==5:
            variable='thetao-130'
        for k in np.arange(0,5):
            month=months[k]
            errors=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_errors_{month+1}.txt')
            x = errors.size // (90 * 160)
            a=errors.reshape((x,90,160))
            
            if month==0:
                lat0=lat.data
                coslat=np.cos(np.deg2rad(lat0))
                weight=np.sqrt(coslat)[...,np.newaxis]                  # latitude weight
                print(weight.shape)   #(90,1) √
                solver = Eof(errors, weights=weight)                   #? Create EOF solver
                EOF= solver.eofsAsCorrelation(neofs=3)             # spatial modes
                PC = solver.pcs(npcs=3, pcscaling=1)                 # time series; pcscaling: Scaling of time series -- choose 0,1,2
                VAR=solver.varianceFraction(neigs=3)

                print(VAR)
                pc1=PC[:,0]
                pc2=PC[:,1]
                pc3=PC[:,2]
                pcx=[pc1,pc2,pc3]
                for l in np.arange(0,3):

                    arr=pcx[l]   #?-----

                    positive_values = arr[arr > 0]
                    negative_values = arr[arr < 0]
                    positive_mean = positive_values.mean()
                    negative_mean = negative_values.mean()
                    positive_indic = np.where(arr > positive_mean)[0]
                    negative_indic = np.where(arr < negative_mean)[0]
                    if l==0:
                        positive_indic1=positive_indic
                        negative_indic1=negative_indic
                    if l==1:
                        positive_indic2=positive_indic
                        negative_indic2=negative_indic
                    if l==2:
                        positive_indic3=positive_indic
                        negative_indic3=negative_indic

            #? 挑选正（负）高度相关的初始误差，并进行合成-------------
            composition=np.empty((6,90,160))
            ppn=np.empty((6,90,160))
            for i in np.arange(0,3):
                if i==0:
                        positive_indices=positive_indic1
                        negative_indices=negative_indic1
                if i==1:
                        positive_indices=positive_indic2
                        negative_indices=negative_indic2
                if i==2:
                        positive_indices=positive_indic3
                        negative_indices=negative_indic3

                ix=2*i
                composition[ix]=errors[positive_indices].mean(axis=0)          #?合成-----
                composition[ix+1]=errors[negative_indices].mean(axis=0)
                t,ppn[ix]=ttest_1samp(errors[positive_indices],0)                    #? significant test
                t,ppn[ix+1]=ttest_1samp(errors[negative_indices],0)

            
            print(np.nanmax(composition))   #方便设置colorbar

            with open(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_composition_{month+1}.txt', 'w') as f:
                # 遍历数组的每个切片
                for two_d_slice in composition:      #?--- 合成的误差（高度相关）
                    np.savetxt(f, two_d_slice, fmt='%1.4f', delimiter=' ')
                    f.write('\n')

            with open(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_ppn_{month+1}.txt', 'w') as f:
                # 遍历数组的每个切片
                for two_d_slice in ppn:      #?--- 显著性检验
                    np.savetxt(f, two_d_slice, fmt='%1.4f', delimiter=' ')
                    f.write('\n')


            #? plot: initial error structure------------------
            fig=plt.figure(figsize=(20,10),dpi=100) #

            #plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
            plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

            cmap1=cmaps.BlueDarkRed18                          #BlueDarkRed18 temp_diff_18lev 
            list_cmap1=cmap1(np.linspace(0,1,10))  
            cp=ListedColormap(list_cmap1,name='cp')

            zt=[231,234,232,235,233,236]
            an=['1+','1-','2+','2-','3+','3-']  #annotate
            for i in np.arange(0,6):
                ztn=zt[i]

                significant_mask=ppn[i,:,:]<0.05
                masked_data = np.where(significant_mask, composition[i,:,:], np.nan) #?-----显著性筛选

                proj=ccrs.PlateCarree(central_longitude=180)                                            
                ax=fig.add_subplot(ztn, projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
                ax.contourf(lon, lat, masked_data , cmap=cp,transform=ccrs.PlateCarree(),levels=np.linspace(-3, 3, 11), extend="both")#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
                ax.add_feature(cfeature.LAND, facecolor='white')
                ax.set_extent([120, 280, -20, 70], crs=ccrs.PlateCarree())                                 
                ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
                ax.tick_params(size=6, labelsize=20)
                #ax.set_xlabel('longitude',size=18)
                #ax.set_ylabel('latitude',size=18)
                ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.set_xticks([180,150,120,-90,-120,-150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
                ax.set_yticks( np.arange(-20,71,20), crs=ccrs.PlateCarree())
                ax.annotate(f'EOF{an[i]}',xy=(0.45,1.1),xycoords='axes fraction',c='k',fontsize=20,ha='left',va='top')

            #绘制colorbar----------------------------------------
            ax0=fig.add_axes([0.2,0.06,0.6,0.03])
            norm =mpl.colors.Normalize(vmin=-3, vmax=3)
            fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                            cmap=cp),cax=ax0,
                            orientation='horizontal',extend='both')
            fc1.set_label('',  rotation=0, labelpad=5, fontsize=20,loc='center')
            ax1=fc1.ax                                         #调用colorbar的ax属性
            #ax1.set_title('Temperature',pad=8, fontsize=16)
            fc1.set_ticks(np.linspace(-3, 3, 11))
            fc1.ax.tick_params(labelsize=20)      #调用colorbar的ax属性

            #子图位置和标题-------------------------------------
            plt.subplots_adjust(left=0.060,
                                bottom=0.150,
                                right=0.970,
                                top=0.92,
                                wspace=0.3,      #子图间水平距离
                                hspace=0.15     #子图间垂直距离
                            )
            plt.suptitle(f'{mdl}_{variable}_errors_{month+1}', fontsize=24, x=0.5, y=0.98)# x,y=0.5,0.98(默认)
            plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_errors_{month+1}.png')
            #plt.show()
            #?==========================================>
        
        #plt.show()#block=False
        #plt.pause(2)  # 显示图像2秒
        plt.close('all')

    # u和v的结果都有了，但循环中只能 处理u或v，我需要的是u和v
    # 也许，我可以在循环外覆盖之前的结果, u+v+t一起处理
    mons=[0,2,5,8,11]
    for kk in np.arange(0,1):
        mon=mons[kk]
        u_comp=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/ua_composition_{mon+1}.txt')
        u_c=u_comp.reshape((6,90,160))
        v_comp=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/va_composition_{mon+1}.txt')
        v_c=v_comp.reshape((6,90,160))
        t_comp=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/tos_composition_{mon+1}.txt')
        t_c=t_comp.reshape((6,90,160))
        t_ppn=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/tos_ppn_{mon+1}.txt')
        t_p=t_ppn.reshape((6,90,160))

        #print(u_c[:,:,60])
        #print(u_c[:,:,61])
        #u_c[:,:,60]=u_c[:,:,61]
        #v_c[:,:,60]=v_c[:,:,61]

        fig=plt.figure(figsize=(20,10),dpi=100) #

        #plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
        plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

        cmap1=cmaps.BlueDarkRed18                          #BlueDarkRed18 temp_diff_18lev 
        list_cmap1=cmap1(np.linspace(0,1,10))  
        cp=ListedColormap(list_cmap1,name='cp')

        zt=[231,234,232,235,233,236]
        an=['1+','1-','2+','2-','3+','3-']  #annotate
        for i in np.arange(0,6):
            ztn=zt[i]

            significant_mask=t_p[i,:,:]<0.05
            masked_data = np.where(significant_mask, t_c[i,:,:], np.nan) #?-----显著性筛选

            proj=ccrs.PlateCarree(central_longitude=180)                                            
            ax=fig.add_subplot(ztn, projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
            
            ax.contourf(lon, lat, masked_data , cmap=cp,transform=ccrs.PlateCarree(),levels=np.linspace(-3, 3, 11), extend="both")#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
            ax.streamplot(lon,lat, u_c[i,:,:], v_c[i,:,:], density=1.3,linewidth=1.3,#scale=50,
                      transform=ccrs.PlateCarree(),color='g'
                       #headwidth=3,headlength=5,headaxislength=4.0
                      )
            ocean_mask = cfeature.NaturalEarthFeature(category='physical', name='ocean', scale='50m', facecolor='none')
            ax.add_feature(ocean_mask, facecolor='none', edgecolor='none')
            #ax.add_feature(cfeature.LAND, facecolor='white')
            ax.add_feature(cfeature.LAND, facecolor='white', zorder=2) #通过 zorder=2 确保陆地部分覆盖在数据之上
            ax.set_extent([120, 280, -20, 70], crs=ccrs.PlateCarree())                                 
            ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
            ax.tick_params(size=6, labelsize=20)
            #ax.set_xlabel('longitude',size=18)
            #ax.set_ylabel('latitude',size=18)
            ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.set_xticks([180,150,120,-90,-120,-150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
            ax.set_yticks( np.arange(-20,71,20), crs=ccrs.PlateCarree())
            ax.annotate(f'EOF{an[i]}',xy=(0.45,1.1),xycoords='axes fraction',c='k',fontsize=20,ha='left',va='top')

        #绘制colorbar----------------------------------------
        ax0=fig.add_axes([0.2,0.06,0.6,0.03])
        norm =mpl.colors.Normalize(vmin=-3, vmax=3)
        fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                        cmap=cp),cax=ax0,
                        orientation='horizontal',extend='both')
        fc1.set_label('',  rotation=0, labelpad=5, fontsize=20,loc='center')
        ax1=fc1.ax                                         #调用colorbar的ax属性
        #ax1.set_title('Temperature',pad=8, fontsize=16)
        fc1.set_ticks(np.linspace(-3, 3, 11))
        fc1.ax.tick_params(labelsize=20)      #调用colorbar的ax属性

        #子图位置和标题-------------------------------------
        plt.subplots_adjust(left=0.060,
                            bottom=0.150,
                            right=0.970,
                            top=0.92,
                            wspace=0.3,      #子图间水平距离
                            hspace=0.15     #子图间垂直距离
                        )
        plt.suptitle(f'{mdl}_uvt_errors_{mon+1}', fontsize=24, x=0.5, y=0.98)# x,y=0.5,0.98(默认)
        plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/uvt_errors_{mon+1}.png')
        