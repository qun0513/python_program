import numpy as np
import xarray as xr
import scipy   
from eofs.standard import Eof
import matplotlib.pyplot as plt
import cartopy.crs as ccrs #crs即坐标参考系统（Coordinate Reference Systems）
import cartopy.feature as cfeature  #添加地图特征，如国界、海岸线等
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter 
 
#忽略警告信息
import warnings
warnings.filterwarnings('ignore')
 
 
#文件读取

dataset = xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
 
sst = dataset['sst']   
sst_loc = sst.loc['1900-01-01':'2018-12-01'].loc[:,70:20,110:260]
ssta_loc=sst_loc.groupby('time.month')-sst_loc.groupby('time.month').mean('time', skipna=True)
 
sst_all = sst.loc['1900-01-01':'2018-12-01']
ssta_all = sst_all.groupby('time.month')-sst_all.groupby('time.month').mean('time', skipna=True)
ssta_all=np.array(ssta_all)
 
ssta_detrend = np.empty((1476,26,76))
for time in np.arange(1476):
    for lat in np.arange(26):
        for lon in np.arange(76):
            ssta_detrend[time,lat,lon]=np.nanmean(ssta_all[time,:,:])
 
ssta_loc=np.array(ssta_loc)
ssta = ssta_loc-ssta_detrend
 
#计算纬度权重
Lat = sst_loc.lat[:]
lat=np.array(Lat)
coslat=np.cos(np.deg2rad(lat))  
wgts = np.sqrt(coslat)[:, np.newaxis]  
 
solver=Eof(ssta,weights=wgts)
eof=solver.eofsAsCorrelation(neofs=3)*(-1)   #空间模态
pc=solver.pcs(npcs=3,pcscaling=1)*(-1)       #时间模态
var=solver.varianceFraction(neigs=3)    #方差贡献率
 
#绘图自定义函数
def mapart(ax):
    
    ax.add_feature(cfeature.COASTLINE,color="k",lw=0.5)  #绘制海岸线  k:黑色  lw为线宽
    ax.add_feature(cfeature.LAND,facecolor="white") #添加陆地  facecolor同color
 
    leftlon,rightlon,lowerlat,upperlat=(110,260,20,70)   #设置经纬范围
    ax.set_extent([leftlon,rightlon,lowerlat,upperlat],crs=ccrs.PlateCarree()) 
    ax.set_xticks(np.arange(leftlon,rightlon,20),crs=ccrs.PlateCarree())    
    ax.set_yticks(np.arange(lowerlat,upperlat+5,10),crs=ccrs.PlateCarree())
    
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
 
 
def eof_contourf(eofs,pcs,pers):
#该函数绘制前三个模态的EOF和PC   绘图三部曲，画布，投影，子图  
    plt.close 
    fig=plt.figure(figsize=(18,10))   #添加画布 figsize=(宽，高)  
    proj=ccrs.PlateCarree(central_longitude=180)  #设置投影类型，并指定中央经线
    lon=np.arange(110,262,2)
    lat=np.arange(70,18,-2)
  
    ax1=fig.add_subplot(321,projection=proj) #添加三行两列子图中第一个子图,即EOF1
    mapart(ax1)
    p=ax1.contourf(lon,lat,eofs[0,:,:],levels=np.arange(-0.9,1.0,0.1), zorder=0, extend = 'both',transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
    ax1.set_title('mode1 (%s'%(round(pers[0],2))+"%)",loc ='left')
 
 
    ax2=fig.add_subplot(322)  #添加三行两列子图中第二个子图,即PC1
    ax2.plot(np.arange(1900,2023,1/12),pcs[:,0],color="k",linewidth=2,linestyle="--")  
    ax2.set_title('pc1 (%s'%(round(pers[0],2))+"%)",loc ='left')
    
    ax3 = fig.add_subplot(323, projection=proj)  #添加三行两列子图中第三个子图,即EOF2
    mapart(ax3)
    pp = ax3.contourf(lon,lat,eofs[1,:,:],levels=np.arange(-0.9,1.0,0.1), zorder=0, extend = 'both',transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
    ax3.set_title('mode2 (%s'%(round(pers[1],2))+"%)",loc ='left')
    
    ax4 = fig.add_subplot(324)  #添加三行两列子图中第四个子图,即PC2
    ax4.plot(np.arange(1900,2023,1/12),pcs[:,1] ,color='k',linewidth=1.2,linestyle='--')
    ax4.set_title('pc2 (%s'%(round(pers[1],2))+"%)",loc ='left')
    
    ax5 = fig.add_subplot(325, projection=proj)   #添加三行两列子图中第五个子图,即EOF3
    mapart(ax5)
    ppp = ax5.contourf(lon,lat,eofs[2,:,:],levels=np.arange(-0.9,1.0,0.1), zorder=0, extend = 'both',transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
    ax5.set_title('mode3 (%s'%(round(pers[2],2))+"%)",loc ='left')
    
    ax6 = fig.add_subplot(326)  #添加三行两列子图中第六个子图,即PC3
    ax6.plot(np.arange(1900,2023,1/12),pcs[:,2],color='k',linewidth=1.2,linestyle='--')
    ax6.set_title('pc3 (%s'%(round(pers[2],2))+"%)",loc ='left')
    
    #添加0线
    ax2.axhline(y=0,  linewidth=1, color = 'k',linestyle='-')
    ax4.axhline(y=0,  linewidth=1, color = 'k',linestyle='-')
    ax6.axhline(y=0,  linewidth=1, color = 'k',linestyle='-')
    
    #在图下边留白边放colorbar        
    fig.subplots_adjust(bottom=0.1,wspace=0.12,hspace=0.3)  #调整子图位置 subplots_adjust(left=,bottom=,top=,right=,hspace=,wspace=)
    cbar_ax = fig.add_axes([0.25,0.04,0.6,0.015]) #[x0, y0, width, height] x0,y0为左下点在图中坐标 width,height为宽度和高度
    
    c=plt.colorbar(pp, cax=cbar_ax,orientation='horizontal')
    c.ax.tick_params(labelsize=6)  #参数labelsize用于设置刻度线标签的字体大小
 
    plt.savefig('eof_detrend.tif',dpi=600,bbox_inches = 'tight',transparent=True, pad_inches = 0)  #将绘图结果存储成tif图片
    plt.show()
 
eof_contourf(eof,pc,var*100)