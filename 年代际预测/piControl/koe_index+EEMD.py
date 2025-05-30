
#?  暂时不看PDO指数，先看看KOE区域的海温
#? koe_idx + EEMD

import numpy as np
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
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import pearsonr


ds=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
lon=ds.longitude
lat=ds.latitude
#print(ds)
ds['longitude'] = xr.where(ds['longitude'] < 0, ds['longitude'] + 360, ds['longitude'])
ds1 = ds.sortby('longitude')   # 确保经度是递增的
longitude1=ds1.longitude
latitude1=ds1.latitude
print(ds1)
#HadISST -1000 and -1.8 需注意

#? 数据处理
'''
#files = "D:/decadal prediction/data/piControl/NorCPM1/tos*.nc"
files='F:/data/piControl/NorCPM1/tos*.nc'
combined_ds = xr.open_mfdataset(
    paths=files, 
    use_cftime=True,                  #时间解码
    #combine="by_coords",    # 按坐标自动对齐
    #parallel=True,                 # 启用并行读取
    #chunks={"time": 100},   # 分块处理大文件
    #engine="netcdf4"        # 指定引擎
)
target_grid = xe.util.grid_global(1, 1)
regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                         #, filename=None,
                         #periodic=True,                 # 处理经度周期性（如全球网格）
                         ignore_degenerate=True,  # 忽略退化的网格点#???某些网格点可能因为几何形状或数值问题导致插值算法无法正常处理
                         #extrap_method='nearest_s2d'  # 外推方法
                         )
regridderdata1= regridder(combined_ds)

# 转换至: 0-360
regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
lat_1d = regridderdata1['lat'].isel(x=0).squeeze().data    # 提取 NumPy 数组
lon_1d = regridderdata1['lon'].isel(y=0).squeeze().data  # 提取 NumPy 数组
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
ds1 = regridderdata1.sortby('lon')
print(combined_ds,ds1)
'''

sst=ds1.sst.sel(
    time=slice('1980-01-01','2018-01-01'),
)#  .data \ .values  .data返回的是xarray数据的底层表示形式，dask\numpy; values, 返回的只是数组
print(sst)  #type(sst)  

# remove the climatological annual cycle------------------
koe = sst.sel(latitude=slice(50,30),longitude=slice(145,210))  # (30°–50°N, 145°E–150°W)  40:60,145:210
print(koe)
koe_shape=np.array(koe.data.shape)
print(koe_shape)
yr=koe_shape[0]//12
sst_koe=koe.data.reshape((yr, 12, koe_shape[1], koe_shape[2]))#.compute()
sst_ac = sst_koe.mean(axis=0)                                            # 直接计算年循环
sst3 = sst_koe - sst_ac                                                         # 广播减法,无需循环，（向量化操作）
sst3=sst3.reshape(koe_shape[0], koe_shape[1], koe_shape[2])



#? 年变率----------------<

'''
pdo0=sst.sel(latitude=slice(60,50),longitude=slice(205,225))
pdo1=sst.sel(latitude=slice(50,30),longitude=slice(225,240))
pdo2=sst.sel(latitude=slice(30,20),longitude=slice(210,270))
nino3=sst.sel(latitude=slice(85,95),longitude=slice(210,270))
sst_pdo0=pdo0.data.reshape(yr,12,len(pdo0.latitude.data),len(pdo0.longitude.data))
sst_pdo1=pdo1.data.reshape(yr,12,len(pdo1.latitude.data),len(pdo1.longitude.data))
sst_pdo2=pdo2.data.reshape(yr,12,len(pdo2.latitude.data),len(pdo2.longitude.data))
sst_nino3=nino3.data.reshape(yr,12,len(nino3.latitude.data),len(nino3.longitude.data))
koe.data[koe.data==-1000]=np.nan  
koe.data[koe.data==-1.8]=np.nan
sst_pdo0[sst_pdo0==-1000]=np.nan  
sst_pdo0[sst_pdo0==-1.8]=np.nan
sst_pdo1[sst_pdo1==-1000]=np.nan  
sst_pdo1[sst_pdo1==-1.8]=np.nan
sst_pdo2[sst_pdo2==-1000]=np.nan  
sst_pdo2[sst_pdo2==-1.8]=np.nan'''
#sst_pdo0 = sst[:,30:40,205:225].reshape((yr, 12, 10, 20))
#sst_pdo1 =sst[:,40:60,225:240].reshape((yr, 12, 20, 15))
#sst_pdo2 =sst[:,60:70,205:250].reshape((yr, 12, 10, 45))
#sst_nino3 = sst[:,85:95,210:270].reshape((yr, 12, 10, 60))

x1=np.nanmean(sst_koe,axis=(2,3))
x2=np.nanmean(x1,axis=0)
x3=x1-x2[None,:]
x4=np.sqrt(np.nansum(x3**2,axis=0)/yr)

'''
x5=np.nanmean(sst_pdo0,axis=(2,3))
x6=np.nanmean(x5,axis=0)
x7=x5-x6[None,:]
x8=np.sqrt(np.nansum(x7**2,axis=0)/yr)

x9=np.nanmean(sst_pdo1,axis=(2,3))
x10=np.nanmean(x9,axis=0)
x11=x9-x10[None,:]
x12=np.sqrt(np.nansum(x11**2,axis=0)/yr)

x13=np.nanmean(sst_pdo2,axis=(2,3))
x14=np.nanmean(x13,axis=0)
x15=x13-x14[None,:]
x16=np.sqrt(np.nansum(x15**2,axis=0)/yr)

print('SST', x2)
print('std', x4)
print(x6)
print(x8)'''

fig=plt.figure(figsize=(10,8),dpi=100)
x=np.arange(12)
ax1=fig.add_subplot(211)
ax1.plot(x,x2,c='b',label='koe_sst')
#ax1.plot(x,x6,c='r',label='pdo0')
#ax1.plot(x,x10,c='g',label='pdo1')
#ax1.plot(x,x14,c='y',label='pdo2')
ax1.set_xticks(np.arange(0,12))
ax1.set_xlabel('month',size=18)
ax1.set_xticklabels(np.arange(1,13))
ax1.tick_params(size=6, labelsize=18)
plt.legend(fontsize=18,frameon=False)

ax2=fig.add_subplot(212)
ax2.plot(x,x4,c='b',label='koe_std')
#ax2.plot(x,x8,c='r',label='pdo0')
#ax2.plot(x,x12,c='g',label='pdo1')
#ax2.plot(x,x16,c='y',label='pdo2')
#ax2.plot(x,(x4+x8+x12+x16)/4,c='k',label='')
ax2.set_xticks(np.arange(0,12))
ax2.set_xlabel('month',size=18)
ax2.set_xticklabels(np.arange(1,13))
ax2.tick_params(size=6, labelsize=18)

'''ax1=fig.add_subplot(222)
ax1.plot(x,x2)
ax2=fig.add_subplot(224)
ax2.plot(x,x4)
ax3=fig.add_subplot(221)
ax3.plot(x,x6)
ax4=fig.add_subplot(223)
ax4.plot(x,x8)'''

plt.suptitle('NorCPM1',x=0.5,y=0.98,fontsize=18)
plt.subplots_adjust(left=0.100,
                    bottom=0.150,
                    right=0.970,
                    top=0.92,
                    wspace=0.3,      #子图间水平距离
                    hspace=0.3     #子图间垂直距离
                   )
plt.legend(fontsize=18,frameon=False)
plt.show()

#? --------------------------------------------------------------------------->



#去趋势，piControl中，无需此步，sst4
'''
sst_g=sst.reshape((119,12,180,360))
sst_gm=np.nanmean(sst_g, axis=0)
sst_gma=sst_g-sst_gm
# area weight -----
a = 6357000  
pi = np.pi
lat = np.array(ds1.sst[0,0:180,0].latitude.values)     #?-----
#print(lat)
dx = 2*pi*a*np.cos(np.deg2rad(lat))/360  
dy = 2*pi*a/360  
s_grid = (dx[:,np.newaxis]*dy)*np.ones((360,))    #?-----
valid_mask = ~np.isnan(sst_gma.reshape(1428, 180,360)[0])  
total_area = np.sum(s_grid*valid_mask) 
weight = s_grid/total_area
global_means = np.nansum(sst_gma.reshape(1428,180,360)*weight[None,:,:], axis=(1,2))  #区域不一样大小
print(global_means.shape)
sst4 = sst3.reshape(1428, 20, 65) - global_means[:, None, None]  #先增加维度((6000,1,1))，后自动广播((6000,20,65))'''


#? area weight -----
'''
a = 6357000  
pi = np.pi
lat = np.array(ds1.sst[0,40:60,0].latitude.values)     #?-----
print(lat)
dx = 2*pi*a*np.cos(np.deg2rad(lat))/360  
dy = 2*pi*a/360  
s_grid = (dx[:,np.newaxis]*dy)*np.ones((65,))    #?-----
valid_mask = ~np.isnan(sst3[0])  
total_area = np.sum(s_grid*valid_mask) 
weight = s_grid/total_area

#koe_idx=np.nansum(sst3*weight[None,:,:], axis=(1,2)) 

'''

# 月指数
#koe_idx=np.nansum(sst3*weight[None,:,:], axis=(1,2))
koe_idx=np.nanmean(sst3,axis=(1,2))
koe_idx=koe_idx#/np.std(koe_idx)     #[600:1428]#.compute()
np.savetxt('D:/decadal prediction/results/piControl/reanalysis/koe_index_1980-2018_HadISST.txt',koe_idx,fmt='%1.6f')



#? koe指数和PDO指数的相关-------------------------<
'''
koe_idx=koe_idx[312:]#.compute()
pdo_idx=np.loadtxt('D:/decadal prediction/results/PDO10/PDOindex_HadISST1950-2018.txt')
pdo_idx=pdo_idx[0:744]
r,p=pearsonr(koe_idx,pdo_idx)
print('r,p',r,p)  #-0.9208698921806686, -0.8759709110296432

l=len(koe_idx)
ll=len(koe_idx)-60
lll=len(koe_idx)-120
# 5年滑动平均，10年周期
y1=np.empty((ll))
y2=np.empty((ll))
for i in np.arange(30, len(koe_idx)-30):
    y1[i-30]=np.mean(koe_idx[i-30: i+30])  
    y2[i-30]=np.mean(pdo_idx[i-30: i+30])  
r1,p1=pearsonr(y1,y2)  # r: -0.91442178850223, -0.9062293190474227
print('r1,p1',r1,p1)

# 10年滑动平均，20年周期
y3=np.empty((lll))
y4=np.empty((lll))
for i in np.arange(60, len(koe_idx)-60):
    y3[i-60]=np.mean(koe_idx[i-60: i+60])  
    y4[i-60]=np.mean(pdo_idx[i-60: i+60])  
r2,p2=pearsonr(y3,y4)  # r: -0.8757869124963087,-0.9281345806283732
print('r2,p2',r2,p2)

fig=plt.figure(figsize=(8,10),dpi=100)
ax=fig.add_subplot()
x=np.arange(0,l)#[0:1200]
x1=np.arange(0,ll)#[0:1140]
ax.plot(x,koe_idx[0:l],c='r',lw=0.6,label='koe index')
ax.plot(x1+30,y1[0:ll],c='r',lw=1.5,label='5a running mean koe index')
ax.plot(x,pdo_idx[0:l],c='b',lw=0.6,label='PDO index')
ax.plot(x1+30,y2[0:ll],c='b',lw=1.5,label='5a running mean PDO index')
ax.set_xlabel('year',fontsize=20)
plt.xticks(ticks=np.arange(0,l+1,120),labels=np.arange(0,l/12+1,10),size=3,fontsize=20)
plt.yticks(ticks=np.arange(-3,5),size=3,fontsize=20)
#ax.tick_params(labelsize=10)
ax.annotate(f'r:{r:.2f}',xy=(0.7,0.1),xycoords='axes fraction',c='k',fontsize=18,ha='left',va='top')

#子图位置和标题-------------------------------------
plt.subplots_adjust(left=0.060,
                    bottom=0.150,
                    right=0.970,
                    top=0.92,
                    wspace=0.3,      #子图间水平距离
                    hspace=0.15     #子图间垂直距离
                   )
plt.suptitle('HadISST_1950-2018', fontsize=20, x=0.5, y=0.98)# x,y=0.5,0.98(默认)

handles,labels = ax.get_legend_handles_labels()
ax.legend(handles,labels, loc='upper center',bbox_to_anchor=(0.5,1),ncol=2,
            frameon=False,fontsize=20)
#plt.show()
'''
#?-------------------------->



#? EEMD --> 周期--------------------------------------<
'''
for i in np.arange(0,6):
    model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
    mdl=model[i]
    koe_idx=np.loadtxt(f"D:/decadal prediction/results/piControl/{mdl}/koeindex.txt")
    eemd = EEMD(trials = 900,
                            noise_width=0.25,
                            random_seed=None,
                            sift_threshold=0.2,                     # 宽松的SD阈值适应低频信号
                            S_number=6,                             # 严格筛选低频IMF
                            spline_kind='cubic',                   # 更光滑的插值方法:cubic\akima
                            extrema_detection='parabol',   # 镜像延拓处理边界
                            #n_processes=4                        # 并行加速
                            ) 

    x=np.arange(0,len(koe_idx))
    y=koe_idx#.compute()
    IMFs = eemd.eemd(y,x)     # signal, time
    n_imfs = len(IMFs)
    print('n_imfs:',n_imfs)

    #可能出现了负的瞬时频率
    # 1.1  使用零点数方案计算周期--------------------
    tt=[]
    for j in np.arange(0,len(IMFs)-1):
        imfs=IMFs[j]
        kk=0
        for i in np.arange(0,len(imfs)-1):
            if imfs[i]*imfs[i+1]<0:
                kk=kk+1
        print(kk)
        t=len(imfs)*2/kk/12
        tt.append(t)   #periods
    tt.append(0)      #趋势项
    print('tt: ',tt,)


    # 1.2 使用 Hilbert transform method 计算周期------------------------
    def calculate_imf_period(imf, dt=1/12):
        analytic_signal = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi * dt)
        
        # 剔除异常值：仅保留0.01-1 Hz（1-100年周期）
        mask = (instantaneous_frequency > 0.01) & (instantaneous_frequency < 1)
        if np.sum(mask) < 10:  # 有效数据不足时返回NaN
            return np.nan
        valid_freq = instantaneous_frequency[mask]
        return 1 / np.mean(valid_freq)

    periods = [calculate_imf_period(IMFs[i]) for i in range(n_imfs)]
    print('periods:',periods)


    # 2. calculate explained variance ------------------
    def calculate_variance_contribution(imfs):
        svar = []
        for imf in imfs:
            var = np.var(imf)
            svar.append(var)
        var_contributions=svar/sum(svar)*100
        return var_contributions
    var_contrib = calculate_variance_contribution(IMFs)
    print('var_contrib:', var_contrib, sum(var_contrib[:]))

    #? plot -----
    fig=plt.figure(figsize=(8, 10),dpi=100)

    ax=fig.add_subplot(n_imfs + 1, 1, 1)
    ax.plot(x, y, color='r', linewidth=0.6)     # original signal
    #ax.plot(x, pdo_idx, color='b', linewidth=0.6)
    #ax.plot(x1+30, y1, color='k', linewidth=1)     # original signal
    ax.set_xticks([])
    ax.tick_params(labelsize=9, size=3)
    #ax.annotate(f"{ttt:.2f}yr",xy=(1.01,0.3),xycoords='axes fraction',c='k',fontsize=9)

    for i, imf in enumerate(IMFs):
        ax1=fig.add_subplot(n_imfs + 1, 1, i + 2)
        ax1.plot(x, imf, color='b', linewidth=0.8)     # each IMF
        ax1.set_xticks([])
        ax1.tick_params(labelsize=9, size=3)
        ax2=ax1.twinx()
        ax2.set_yticks([])
        ax2.annotate(f"{tt[i]:.2f}yr\n{var_contrib[i]:.2f}",xy=(1.01,0.3),xycoords='axes fraction',c='k',fontsize=9)#,ha='left',va='top')
        #ax2.set_ylabel(f"{tt[i]:.2f}yr\n{var_contrib[i]:.2f}" , size=9, labelpad=45, rotation=0, loc='top')
    plt.xticks(ticks=np.arange(0, 6000+1, 600), labels=[f"{i}" for i in np.arange(0, 500+1, 50)], size=9)
    ax1.set_xlabel('year', size=9)

    # Subgraphs position and title -------------------------------------
    plt.subplots_adjust(left=0.10,
                        bottom=0.060,
                        right=0.900,
                        top=0.955,
                        wspace=0.18,                 # horizontal distance between subgraphs
                        hspace=0.300                # vertical distance between subgraphs
                    )
    plt.suptitle(f'EEMD_koeindex_{mdl}', fontsize=12, x=0.5, y=0.98)   # x,y=0.5,0.98 (default)

    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/EEMD_koeindex_{mdl}.png')
    #plt.show()
'''
#?------------------------------->