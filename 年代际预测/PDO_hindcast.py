import numpy as np
import xarray as xr
from scipy.stats import zscore
from eofs.standard import Eof
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
import matplotlib as mpl
from eofs.standard import Eof
import xskillscore as xs
import xesmf as xe
from scipy.stats import pearsonr
#import cf
#import cfplot as cfp
#忽略警告信息
import warnings
warnings.filterwarnings('ignore')

import os
import glob
#数据处理---------------------------------------------------------------------------------------------------------------------
#data1=pd.read_table('D:/decadal prediction/data/ersst.v5.pdo.dat.txt',sep=r'\s+',header=2)
#pdo0=np.array(data1.values)   #values、data
#pdo1=pdo0[154:164,1:13]
#pdo2=pdo1.reshape(120)

#MSSS=[]
#ACC=[]
def f(year):

    fig=plt.figure(figsize=(25,10),dpi=200)  #每年一张图

    directory = "D:\decadal prediction\data\hindcast\MIROC6"
    file_pattern = f"tos_Omon_MIROC6_dcppA-hindcast_s{year}*.nc"
    file_paths = glob.glob(os.path.join(directory, file_pattern))  # glob 模块提供了一种方便的方式来使用 Unix shell 风格的通配符匹配文件路径

    #eof=np.loadtxt(f"D:/decadal prediction/results/PDOpattern_HadISST_{year+1}-{year+1+10}.txt")
    #Msss=[ ]
    #Acc=[ ]
    #print(file_paths)
    for r in np.arange(0,10):
        data=xr.open_dataset(file_paths[r])
        
        target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
        regridder = xe.Regridder(data, target_grid, 'bilinear', filename=None)
        regridderdata= regridder(data['tos'])
        #print(regridderdata.shape)
        rdata=np.empty(regridderdata.shape)                                               #（-180,180） 转为（0，360）------------------
        for i in np.arange(0,360):
            if i<180:
                rdata[:,:,i]=regridderdata[:,:,180+i]
            if i>=180:
                rdata[:,:,i]=regridderdata[:,:,i-180]
        lon=regridderdata.lon[110:160,110:260]                          #  也很关键，要理解数据定位的区域 和 画布定位的区域，如何让二者重合
        lat=regridderdata.lat[110:160,110:260]                            #    经纬度都挑选那个区域
        #print(lon,lat)
        #rdata[:,:,180]=rdata[:,:,181]                                               #循环补全
        # remove the global-mean SST anomaly----------------------
        
        sst1=rdata[2:122,:,:]
        sst_1=rdata[0:2,:,:]
        #print(number, sst1.shape, sst_1.shape)

        # remove the global-mean SST anomaly----------------------
        sst_g=sst1.reshape((10,12,180,360))
        sst_gm=np.nanmean(sst_g, axis=0)
        sst_gm[10,:,:]=sst_gm[10,:,:]*10/11+sst_1[0,:,:]/11   #11月
        sst_gm[11,:,:]=sst_gm[11,:,:]*10/11+sst_1[1,:,:]/11   #12月

        sst_gma0=np.empty((10,12,180,360))
        for j in np.arange(0,12):
            for k in np.arange(0,180):
                for l in np.arange(0,360):
                    sst_gma0[:,j,k,l]=sst_g[:,j,k,l]-sst_gm[j,k,l]
        sst_gma0=sst_gma0.reshape((120,180,360))
        sst_gma1=sst_1[0:2,:,:]-sst_gm[10:12,:,:]
        sst_gma=np.concatenate((sst_gma1,sst_gma0))  #122


        #remove the climatological annual cycle------------------------
        sst_np=sst1[:,110:160,110:260]                          #北太平洋海温   (1791,50,150)
        sst_nps=sst_np.reshape((10,12,50,150))        #年循环
        sst_ac=np.nanmean(sst_nps,axis=0)             #年循环气候态
        sst_ac[10,:,:]=sst_ac[10,:,:]*10/11+sst_1[0,110:160,110:260]/11
        sst_ac[11,:,:]=sst_ac[11,:,:]*10/11+sst_1[1,110:160,110:260]/11

        sst2_1=np.empty((120,50,150))
        for i in np.arange(120):
            for j in np.arange(0,50):
                for k in np.arange(0,150):
                    sst2_1[i,j,k]=sst_np[i,j,k] -np.nanmean( sst_gma[i,:,:]  )                   #(1788,50,150)
        sst2_0=sst_1[0:2,110:160,110:260]-np.nanmean( sst_gma[10:12,:,:]  )
        sst2=np.concatenate((sst2_0,sst2_1))   #122

        sst3_1=np.empty((10,12,50,150))
        for j in np.arange(0,12):
            for k in np.arange(0,50):
                for l in np.arange(0,150):
                    sst3_1[:,j,k,l]=sst2[2:122,:,:].reshape((10,12,50,150))[:,j,k,l]-sst_ac[j,k,l]
        #sst=sst3.reshape((120,50,150))
        sst3_0=sst2[0:2,:,:]-sst_ac[10:12,:,:]
        sst3=np.concatenate((sst3_0,sst3_1.reshape((120,50,150))))
        sst=sst3

        #10年，120个时间节点
        '''
        sst1=rdata[2:122,:,:]
        
        sst_g=sst1.reshape((10,12,180,360))
        sst_gm=np.nanmean(sst_g,axis=0)
        sst_gma=np.empty((10,12,180,360))
        for j in np.arange(0,12):
            for k in np.arange(0,180):
                for l in np.arange(0,360):
                    sst_gma[:,j,k,l]=sst_g[:,j,k,l]-sst_gm[j,k,l]
        sst_gma=sst_gma.reshape((120,180,360))

        #remove the climatological annual cycle------------------------
        sst_np=sst1[:,110:160,110:260]                          #北太平洋海温   (1791,50,150)
        sst_nps=sst_np.reshape((10,12,50,150))            #年循环
        sst_ac=np.nanmean(sst_nps,axis=0)                 #年循环气候态

        sst2=np.empty((120,50,150))
        for i in np.arange(120):
            for j in np.arange(0,50):
                for k in np.arange(0,150):
                    sst2[i,j,k]=sst_np[i,j,k] -np.nanmean( sst_gma[i,:,:]  )                   #(1788,50,150)

        sst3=np.empty((10,12,50,150))
        for j in np.arange(0,12):
            for k in np.arange(0,50):
                for l in np.arange(0,150):
                    sst3[:,j,k,l]=sst2.reshape((10,12,50,150))[:,j,k,l]-sst_ac[j,k,l]
        sst=sst3.reshape((120,50,150))
        '''

        #原数据的EOF
        #print(regridderdata.lat.loc[160:110,0])
        lat0=lat.data[:,0]
        #print(lat0.shape)
        coslat=np.cos(np.deg2rad(lat0))
        weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重

        solver = Eof(sst, weights=weight)                        # 创建EOF求解器
        EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
        PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
        VAR=solver.varianceFraction(neigs=3)
        EOF[0,:,70]=EOF[0,:,71]                                        #循环补全，相同的效果

        #np.savetxt(f"D:/decadal prediction/results/pdoindex_hindcast_{year+1}_r{r+1}.txt",PC[:,0],fmt='%1.8f')
        #np.savetxt(f"D:/decadal prediction/results/pdopattern_hindcast_{year+1}_r{r+1}.txt",EOF[0,:,:],fmt='%1.8f')
        
        pdoi0=np.loadtxt(f"D:/decadal prediction/results/PDOindex_HadISST_{year+1}-{year+1+10}.txt")       #观测
        pdop0=np.loadtxt(f"D:/decadal prediction/results/PDOpattern_HadISST_{year+1}-{year+1+10}.txt")   #观测
        
        eof=xr.DataArray(EOF[0,:,:].reshape((7500)),dims=['time'])
        pdop=xr.DataArray(pdop0.reshape((7500)),dims=['time'])
        
        
        for i in np.arange(30, len(PC[:,0])-30):
            pdoi0[i]=np.mean(pdoi0[i-30:i+30])
            PC[i,0]=np.mean(PC[i-30:i+30,0])
        
        rp=xs.pearson_r(eof,pdop,dim='time',skipna=True)
        ri,p=pearsonr(PC[:,0], pdoi0)
        rt0=rp*ri
        rt=float(f"{rt0.data:.3f}")

        print( year+1, r+1, rp.data, ri, rt)

        '''
        #投影的EOF----------------
        print(sst.reshape(sst.shape[0],-1))
        print(eof.reshape(-1))
        pc = np.dot(np.isnan(sst.reshape(sst.shape[0],-1)), np.isnan(eof.reshape(-1)))       #预测
        print('pc',pc)
        np.savetxt(f"D:/decadal prediction/results/PDOindex_{year+1}-{year+1+10}hindcast{r}.txt",pc)
        obs=np.loadtxt(f"D:/decadal prediction/results/PDOindex_HadISST_{year+1}-{year+1+10}.txt")   #观测

        # MSSS
        mean_observation = np.mean(obs)
        forecast_error_sum = np.sum((pc - obs) ** 2)
        print(forecast_error_sum)
        observation_variance = np.sum((obs - mean_observation) ** 2)
        print(observation_variance)
        msss = 1 - (forecast_error_sum / observation_variance)
        print(msss)
        Msss.append(msss)

        # ACC
        mean_forecast = np.mean(pc)
        mean_observation = np.mean(obs)
        f_dev = pc - mean_forecast
        a_dev = obs - mean_observation
        numerator = np.sum(f_dev * a_dev)
        print(numerator)
        denominator = np.sqrt(np.sum(f_dev**2) * np.sum(a_dev**2))
        print(denominator)
        acc = numerator / denominator
        print(acc)
        Acc.append(acc)
        '''

        #np.savetxt(f"pdoindex_hindcast_{year}r{r+1}.txt",PC[:,0],fmt='%1.8f')
        #np.savetxt(f"pdopattern_hindcast_{year}r{r+1}.txt",EOF[0,:,:],fmt='%1.8f')
        #pdo=pd.Series(pdo2)
        #pc1=pd.Series(pc)
        #cc=pdo.corr(pc1,method='pearson')                    #相关性
        #print(f'{year}r{r+1}cc',cc,'\n')
        '''
        将您的目标三维数据投影到第一模态上，实际上是计算目标数据与第一模态之间的点积。这会给出每个时间步的主成分得分--时间系数。
        projection = np.dot(target_data, first_eof)
        '''

        
        #绘图-------------------------------------------------------------------------------------
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']         # 用来正常显示中文标签 SimHei FangSong
        plt.rcParams['axes.unicode_minus'] = False                     # 用来正常显示负号
        cmap1=cmaps.BlueDarkRed18                                        # BlueDarkRed18 temp_diff_18lev
        list_cmap1=cmap1(np.linspace(0,1,20))
        cp=ListedColormap(list_cmap1,name='cp')

        rr=[1,2,3,4,5,6,7,8,10,11]  
        #subp=(3,4,r+1)
        proj=ccrs.PlateCarree(central_longitude=180)                         #!   central_longitude=180，只用给子图属性一次即可，不然(可能)后面会重置
        ax=fig.add_subplot(3,4,rr[r],projection=proj)                           # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
        ax.contourf(lon, lat, EOF[0,:,:] , cmap=cp, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
        #ax.set_extent([-180,180,-90,90],crs=ccrs.PlateCarree() )         #不如 ‘数据+投影’  直接定位 -->北太平洋区域--这个范围不好设置
        ax.coastlines(resolution="50m", linewidth=0.8)                       # resolution 分辨率（valid scales）--110m、50m、10m。
        ax.tick_params(size=3, labelsize=12)
        ax.set_xlabel('longitude',size=12)
        ax.set_ylabel('latitude',size=12)
        ax.set_title(f'PDO_hindcast_{year+1}r{r+1}({rt})')
        
        #画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
        #但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.
        
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks([180,150,120,-120,-150,-180],crs=ccrs.PlateCarree())      #似乎很关键√√ ticks始终显示的是lon,lat的值（上两行可修正），但就看显示的是谁的点位                              # [ ] 有特定经度就行，不在乎顺序
        ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())                  # 也能定画图范围,crs可略去
        

    '''
    #时间序列-------------------------------------------
    ax1=fig.add_subplot(212)
    ax1.plot(np.arange(0,1788),PC[:,0],c='b',lw=1)
    #ax1.plot(np.arange(0,1788),df,c='k',lw=3)
    ax1.set_xticks([0,359,719,1079,1439,1767])
    #ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax1.set_xlabel('year')
    ax1.set_xticklabels([1870,1900,1930,1960,1990,2018])
    ax1.tick_params(size=6, labelsize=18)
    '''

    
    #绘制colorbar----------------------------------------
    ax0=fig.add_axes([0.20,0.08,0.6,0.02])
    norm =mpl.colors.Normalize(vmin=-1, vmax=1)
    fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cp),cax=ax0,
                                 orientation='horizontal',extend='both')
    fc1.set_label('Temperature',  rotation=0, labelpad=4, fontsize=15,loc='center')
    ax1=fc1.ax                                         #调用colorbar的ax属性
    #ax1.set_title('Temperature',pad=8, fontsize=16)
    fc1.set_ticks(np.linspace(-1, 1, 11))
    fc1.ax.tick_params(size=3,labelsize=12)      #调用colorbar的ax属性

    #子图位置和标题-------------------------------------
    plt.subplots_adjust(left=0.040,
                        bottom=0.120,
                        right=0.980,
                        top=0.980,
                        wspace=0.18,                 #子图间水平距离
                        hspace=0.018                #子图间垂直距离
                    )
    #plt.suptitle(f'PDO_{year}_hindcast', fontsize=18, x=0.5, y=0.965)# x,y=0.5,0.98(默认)
    #plt.show()
    #plt.savefig(f'D:/decadal prediction/results/PDO_hindcast_{year+1}.jpg')
    
    #MSSS.append(Msss)
    #ACC.append(Acc)
    #np.savetxt(f'D:/decadal prediction/results/MSSS{year+1}.txt',Msss,fmt='%1.8f')
    #np.savetxt(f'D:/decadal prediction/results/ACC{year+1}.txt',Acc,fmt='%1.8f')
#并行----------------------------------------------
from multiprocessing import Pool
import time

if __name__ == '__main__':
    start_time = time.time()
    with Pool(1) as p:
        # 如果您需要收集返回值，可以使用 p.starmap 并提供一个包含参数和函数的元组列表
        # p.starmap(f, [(i,) for i in np.arange(0, 3)])
        p.map(f, np.arange(1960,1961))  #1960-2009
    end_time = time.time()
    print('Takeing :', end_time - start_time)






    

# better comments--------------------------------------------------------------------------------
#//! 红色的高亮注释
#//? 蓝色的高亮注释
#//* 绿色的高亮注释
#//todo 橙色的高亮注释
#// 灰色带删除线的注释

'''
# 创建新的经纬度坐标数组
new_lons = np.arange(-179.5, 180.5)
new_lats = np.arange(-89.5, 90.5, 1)
# 使用 assign_coords 方法替换现有的坐标
ds_new_coords = rdata.assign_coords(lon=new_lons, lat=new_lats)
'''