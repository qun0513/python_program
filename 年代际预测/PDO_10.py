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
from multiprocessing import Pool
from scipy.stats import pearsonr
import xskillscore as xs
#忽略警告信息
import warnings
warnings.filterwarnings('ignore')

#数据处理---------------------------------------------------------------------------------------------------------------------
data=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
#data1=pd.read_table('D:/decadal prediction/data/ersst.v5.pdo.dat.txt',sep=r'\s+',header=2)
#pdo0=np.array(data1.values)
#pdo1=pdo0[154:164,1:13]
#pdo2=pdo1.reshape(120)

def f(year):       #从11月开始分解PDO
    fig=plt.figure(figsize=(30,10),dpi=200) #
    numb=year-1870     #某一年后的10年
    for v in np.arange(0,10):
        number=numb+v
        sst0=np.empty((122,180,360))      #（-180,180） 转为（0，360）------------------
        for i in np.arange(0,360):
            if i<180:
                sst0[:,:,i]=data.sst[number*12-2: (number+10)*12,:,180+i]
            if i>=180:
                sst0[:,:,i]=data.sst[number*12-2: (number+10)*12,:,i-180]
        sst0[sst0==-1000]=np.nan
        sst0[sst0==-1.8]=np.nan
        
        sst1=sst0[2: 122,:,:]
        sst_1=sst0[0:2,:,:]
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
        sst_np=sst1[:,20:70,110:260]                          #北太平洋海温   (1791,50,150)
        sst_nps=sst_np.reshape((10,12,50,150))        #年循环
        sst_ac=np.nanmean(sst_nps,axis=0)             #年循环气候态
        sst_ac[10,:,:]=sst_ac[10,:,:]*10/11+sst_1[0,20:70,110:260]/11
        sst_ac[11,:,:]=sst_ac[11,:,:]*10/11+sst_1[1,20:70,110:260]/11

        sst2_1=np.empty((120,50,150))
        for i in np.arange(120):
            for j in np.arange(0,50):
                for k in np.arange(0,150):
                    sst2_1[i,j,k]=sst_np[i,j,k] -np.nanmean( sst_gma[i,:,:]  )                   #(1788,50,150)
        sst2_0=sst_1[0:2,20:70,110:260]-np.nanmean( sst_gma[10:12,:,:]  )
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


        lat0=data.sst.latitude.loc[70:20].data
        #print(lat0)
        coslat=np.cos(np.deg2rad(lat0))
        #print(coslat.shape)
        weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重

        solver = Eof(sst, weights=weight)                        # 创建EOF求解器
        EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
        PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
        VAR=solver.varianceFraction(neigs=3)

        np.savetxt(f'D:/decadal prediction/results/PDOindex_HadISST_{year+v}-{year+v+10}.txt',PC[:,0],fmt='%1.8f')
        np.savetxt(f'D:/decadal prediction/results/PDOpattern_HadISST_{year+v}-{year+v+10}.txt',EOF[0,:,:],fmt='%1.8f')

        #空间相关_10
        pattern10=EOF[0,:,:].reshape((7500))
        pattern0=np.loadtxt("D:/decadal prediction/results/PDOpattern_HadISST1950-2018.txt")
        pattern1=pattern0.reshape((7500))
        pattern1=xr.DataArray(pattern1,dims=['time'])
        pattern10=xr.DataArray(pattern10,dims=['time'])
        r0=xs.pearson_r(pattern1,pattern10,dim='time',skipna=True)
        r0 =float(f"{r0.data:.3f}")
        r1=abs(r0)

        #时间相关_10
        ts0=np.loadtxt("D:/decadal prediction/results/PDOindex_HadISST1950-2018.txt")
        ts1=ts0[(year-1950+v)*12-2:(year-1950+v+10)*12]
        ts1=xr.DataArray(ts1,dims=['time'])
        ts2=xr.DataArray(PC[:,0],dims=['time'])
        r2=xs.pearson_r(ts1,ts2,dim='time',skipna=True)
        r2 =float(f"{r2.data:.3f}")
        r3=abs(r2)
        
        #print(PC[:,0])
        #print(EOF[0,:,:].shape)
        print(VAR)

        #pdo=pd.Series(pdo2)
        #pc1=pd.Series(PC[:,0])
        #r=pdo.corr(pc1,method='pearson')                    #相关性
        #print('r','\n',r)

        #df = pd.DataFrame(pdo2, columns=['Value'])
        #df['5Y_Moving_Average'] = df['Value'].rolling(window=60).mean()    #滑动平均

        lon=data.sst.longitude[110:260]                          #!  也很关键，要理解数据定位的区域 和 画布定位的区域，如何让二者重合
        lat=data.sst.latitude[20:70]
        #lon=np.arange(110,260)
        #lat=np.arange(20,70)
        #lon,lat=np.meshgrid(lon,lat)

        """
        from cartopy.util import add_cyclic_point
        sst,lon=add_cyclic_point(sst, coord=lon)  #循环补全数据；最后一个由下一循环的第一个补上
        sst[:,180] = sst[:, 181]    #
        """
        EOF[0,:,70]=EOF[0,:,71]                                        #循环补全，相同的效果
        
        #绘图------------------------------------------------------------------------------------------------------------------------
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
        plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

        cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
        list_cmap1=cmap1(np.linspace(0,1,20))
        cp=ListedColormap(list_cmap1,name='cp')
        
        proj=ccrs.PlateCarree(central_longitude=180)                                            #!   central_longitude=180，只用给子图属性一次即可，不然(可能)后面会重置
        rr=[1,2,3,4,5,6,7,8,10,11]
        ax=fig.add_subplot(3,4,rr[v],projection=proj)                                                          # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
        #plt.sca(ax)
        ax.contourf(lon, lat, EOF[0,:,:], cmap=cmap1)#,crs=ccrs.PlateCarree())  #, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
        #ax.set_extent([110,150,20,70],crs=ccrs.PlateCarree() )                               #!不如 ‘数据+投影’  直接定位 --北太平洋区域
        ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
        ax.tick_params(size=2, labelsize=6)
        ax.set_xlabel('',size=6)
        ax.set_ylabel('',size=6)
        ax.set_title(f'PDO_{year+v}-{year+v+10}({r0}/{r2})',size=8)
        #? 画布已经定位了中心经度，后只需将数据对应于中心经度的点，投影上来即可；
        #? 但需注意，后续只需在此基础上操作即可，无需带上一遍中心经度，不然会重置前面的基础，导致混乱.

        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks([180,150,120,-120,-150,-180],crs=ccrs.PlateCarree())     # [ ] 有特定经度就行，不在乎顺序   
        ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())                 #!!? 显示的始终是lon和lat的值，但就看显示在谁的点位上了；这个位置的crs也很关键
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
    ax0=fig.add_axes([0.2,0.08,0.6,0.03])
    norm =mpl.colors.Normalize(vmin=-1, vmax=1)
    fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                    cmap=cp),cax=ax0,
                    orientation='horizontal',extend='both',fraction=0.12)
    fc1.set_label('Temperature',  rotation=0, labelpad=3, fontsize=6,loc='center')
    ax1=fc1.ax                                         #调用colorbar的ax属性
    #ax1.set_title('Temperature',pad=8, fontsize=16)
    fc1.set_ticks(np.linspace(-1, 1, 11))
    fc1.ax.tick_params(labelsize=10)      #调用colorbar的ax属性

    #子图位置和标题-------------------------------------
    plt.subplots_adjust(left=0.040,
                        bottom=0.160,
                        right=0.980,
                        top=0.96 ,
                        wspace=0.18,      #子图间垂直距离
                        hspace=0.018     #子图间水平距离
                    )
    #plt.suptitle('PDO_10', fontsize=20, x=0.5, y=0.92)# x,y=0.5,0.98(默认)
    #plt.tight_layout()
    plt.show()  # 0: 120+; 1: 120+_; 2: 122+_; 3:122+_ running mean;
    #plt.savefig(f'pdo_{year}-{year+10}.jpg')

#并行----------------------------------------------
from multiprocessing import Pool
import time

if __name__ == '__main__':
    start_time = time.time()
    with Pool(6) as p:
        # 如果您需要收集返回值，可以使用 p.starmap 并提供一个包含参数和函数的元组列表
        # p.starmap(f, [(i,) for i in np.arange(0, 3)])
        p.map(f, np.arange(1951, 2001,10))
    end_time = time.time()
    print('Takeing :', end_time - start_time)







# better comments--------------------------------------------------------------------------------
#//! 红色的高亮注释
#//? 蓝色的高亮注释
#//* 绿色的高亮注释
#//todo 橙色的高亮注释
#// 灰色带删除线的注释