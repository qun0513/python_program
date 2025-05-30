 #?   PDO_500
 #?   EEMD-->singals + periods 
#? 模式之间的修改在输入（读取文件）输出（保存index、pattern、figure）

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


#读取数据------------------
files = "F:/data/piControl/HadGEM3-GC31-LL/tos*.nc"
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

# to: 0-360
regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
lat_1d = regridderdata1['lat'].isel(x=0).squeeze().data  # 提取 NumPy 数组
lon_1d = regridderdata1['lon'].isel(y=0).squeeze().data  # 提取 NumPy 数组
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
ds = regridderdata1.sortby('lon')

#? EOF(Empirical Orthogonal Function)-----------------
tos_np=ds.tos[0:6000,110:160,120:260].data        # 20N-70N, 120E-100W
# remove the climatological annual cycle------------------
sst_nps = tos_np.reshape((500, 12, 50, 140))
sst_ac = sst_nps.mean(axis=0)  
sst3 = sst_nps - sst_ac        
sst=sst3.reshape((6000,50,140))

"""remove the global-mean SST anomaly----------------------
sst_g=sst0.reshape((149,12,180,360))  #1970-2018
sst_gm=np.nanmean(sst_g,axis=0)
sst_gma=sst_g-sst_gm
global_means = np.nanmean(sst_gma.reshape(1788, 180,360), axis=(1,2))
print(global_means.shape)
sst4 = sst3.reshape(1788, 50, 150) - global_means[:, None, None]
"""

lon=ds.tos[0,0,120:260].lon.data
lat=ds.tos[0,110:160,0].lat.data
print(lon)
print(lat)
coslat=np.cos(np.deg2rad(lat))
weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重
print(weight)

solver = Eof(sst, weights=weight)                        # 创建EOF求解器
EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
VAR=solver.varianceFraction(neigs=3)
print('VAR:',VAR)

#todo---------------------------
#np.savetxt("D:/decadal prediction/results/piControl/MRI-ESM2-0/MRI-ESM2-0_PDOindex.txt", PC[:,0], fmt='%1.8f')
#np.savetxt("D:/decadal prediction/results/piControl/MRI-ESM2-0/MRI-ESM2-0_PDOpattern.txt", EOF[0,:,:], fmt='%1.8f')

#? plot -------------------------------------
fig=plt.figure(figsize=(8,10),dpi=100) #

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

cmap1=cmaps.BlueDarkRed18                                #BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,10))  
cp=ListedColormap(list_cmap1,name='cp')
proj=ccrs.PlateCarree(central_longitude=180)                                            
ax=fig.add_subplot(211,projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
contourf_plot=ax.contourf(lon, lat, EOF[0,:,:] , cmap=cp,transform=ccrs.PlateCarree(),
                                           levels=np.linspace(-1, 1, 11), extend="both")#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
ax.add_feature(cfeature.LAND, facecolor='white')
ax.set_extent([120, 260, 20, 70], crs=ccrs.PlateCarree())                                 
ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
ax.tick_params(size=6, labelsize=16)
#ax.set_xlabel('longitude',size=18)
#ax.set_ylabel('latitude',size=18)
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([180,150,120,-120,-150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())
ax.set_title('spatial mode',size=15, pad=10)
#ax.annotate(f'EOF{an[i]}',xy=(0.5,1.1),xycoords='axes fraction',c='b',fontsize=16,ha='left',va='top')

norm =mpl.colors.Normalize(vmin=-1, vmax=1)
colorbar = plt.colorbar(contourf_plot, norm=norm, orientation='vertical', extend='both', shrink=1, pad=0.03)
colorbar.set_label('', fontsize=12)
colorbar.set_ticks(np.linspace(-1, 1, 11))  # 设置刻度位置
colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(-1,1,11)],size=12)  # 设置刻度标签

ax2=fig.add_subplot(212) 
x=np.arange(0,6000)
x1=np.arange(0,5940)
y=PC[:,0]  # [:,0]
y1=np.empty((5940))
for i in np.arange(30, len(y)-30):
    y1[i-30]=np.mean(y[i-30: i+30])
ax2.plot(x, y, c='r', lw=0.3)
ax2.plot(x1, y1, c='k', lw=1.2)
ax2.set_title('time series', size=15, pad=10)
ax2.set_xticks(np.arange(0,6001,600))
ax2.set_xticklabels(np.arange(0,501,50))
ax2.set_xlabel('year',size=15)
ax2.tick_params(size=3, labelsize=15)
#ax2.set_aspect(0.8)

#Subgraphs position and title-------------------------------------
plt.subplots_adjust(left=0.070,
                    bottom=0.080,
                    right=0.950,
                    top=0.890,
                    wspace=0.18,                 # horizontal distance between subgraphs
                    hspace=0.300                # vertical distance between subgraphs
                )
#plt.tight_layout()
plt.suptitle('HadGEM3-GC31-LL_PDO', fontsize=18, x=0.55, y=0.98)   # x,y=0.5,0.98 (default)
#plt.savefig("")
plt.show()

sst=ds.tos.isel(
    time=slice(0,6000),
)#  .data \ .values  .data返回的是xarray数据的底层表示形式，dask\numpy; values, 返回的只是数组
#print(sst)  #type(sst)  
# remove the climatological annual cycle-----
koe = sst.sel(y=slice(120,140),x=slice(145,210))  # (30°–50°N, 145°E–150°W)  40:60,145:210
print(koe)

koe_shape=np.array(koe.data.shape)
yr=koe_shape[0]//12
print(koe_shape)
sst_koe=koe.data.reshape((yr, 12, koe_shape[1], koe_shape[2]))
#sst_koe[sst_koe==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
#sst_koe[sst_koe==-1.8]=np.nan
sst_ac = sst_koe.mean(axis=0)                                            # 直接计算年循环
sst5 = sst_koe - sst_ac                                                         # 广播减法,无需循环，（向量化操作）
sst5=sst5.reshape((koe_shape[0], koe_shape[1], koe_shape[2]))

#sst6=sst5-global_means[:, None, None]
koe_idx=np.nanmean(sst5,axis=(1,2)).compute()

r,p=pearsonr(PC[:,0],koe_idx)
print(r,p)






#? EEMD-------------------------------------------------------------------->
'''
PC=np.loadtxt("D:/decadal prediction/results/piControl/MRI-ESM2-0/MRI-ESM2-0_PDOindex.txt")   #?-----

#  EEMD   (500\200\100 year)   -------------
eemd = EEMD(trials = 400,
                        noise_width=0.2,
                        random_seed=None,
                        sift_threshold=0.2,                     # 宽松的SD阈值适应低频信号
                        S_number=6,                             # 严格筛选低频IMF
                        spline_kind='cubic',                   # 更光滑的插值方法:cubic\akima
                        extrema_detection='parabol',   # 镜像延拓处理边界
                        #n_processes=4                        # 并行加速
                        )                      

x=np.arange(0,6000)
y=PC[0:6000]
x1=np.arange(0,5940)
y1=np.empty((5940))
for i in np.arange(30, len(y)-30):
    y1[i-30]=np.mean(y[i-30: i+30])

IMFs = eemd.eemd(y,x)     # signal, time
n_imfs = len(IMFs)

r,p=pearsonr(IMFs[5][30:5970], y1)
print(r,p) 


# 1. Calculate the IMF cycle using the Hilbert transform method------------------
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

#可能出现了负的瞬时频率
# 1.1  counting cross zero numbers
tt=[]
for j in np.arange(0,len(IMFs)):
    imfs=IMFs[j]
    kk=0
    for i in np.arange(0,len(imfs)-1):
        if imfs[i]*imfs[i+1]<0:
            kk=kk+1
    t=len(imfs)*2/kk/12
    #print('t:', len(imfs)*2/kk)
    tt.append(t)   #periods
print('tt: ',tt,)
kkk=0
for i in np.arange(0,len(y1)-1):
    if y1[i]*y1[i+1]<0:
        kkk=kkk+1
ttt=len(y1)*2/kkk/12
print(ttt)

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


# plot -----------------------------------------------------------------------------------
fig=plt.figure(figsize=(8, 10),dpi=100)

ax=fig.add_subplot(n_imfs + 1, 1, 1)
ax.plot(x, y, color='r', linewidth=0.6)     # original signal
ax.plot(x1+30, y1, color='k', linewidth=1)     # original signal
ax.set_xticks([])
ax.tick_params(labelsize=9, size=3)
ax.annotate(f"{ttt:.2f}yr",xy=(1.01,0.3),xycoords='axes fraction',c='k',fontsize=9)

for i, imf in enumerate(IMFs):
    ax1=fig.add_subplot(n_imfs + 1, 1, i + 2)
    ax1.plot(x, imf, color='b', linewidth=0.8)     # each IMF
    ax1.set_xticks([])
    ax1.tick_params(labelsize=9, size=3)
    ax2=ax1.twinx()
    ax2.set_yticks([])
    ax2.annotate(f"{tt[i]:.2f}yr\n{var_contrib[i]:.2f}",xy=(1.01,0.3),xycoords='axes fraction',c='k',fontsize=9)#,ha='left',va='top')
    #ax2.set_ylabel(f"{tt[i]:.2f}yr\n{var_contrib[i]:.2f}" , size=9, labelpad=45, rotation=0, loc='top')
plt.xticks(ticks=np.arange(0, 6001, 600), labels=[f"{i}" for i in np.arange(0, 501, 50)], size=9)
ax1.set_xlabel('year', size=9)

# Subgraphs position and title -------------------------------------
plt.subplots_adjust(left=0.10,
                    bottom=0.066,
                    right=0.900,
                    top=0.960,
                    wspace=0.18,                 # horizontal distance between subgraphs
                    hspace=0.300                # vertical distance between subgraphs
                )
plt.suptitle('MRI-ESM2-0_PDOindex_EEMD', fontsize=12, x=0.5, y=0.98)   # x,y=0.5,0.98 (default)

plt.savefig("D:/decadal prediction/results/piControl/MRI-ESM2-0/MRI-ESM2-0_PDOindex_EEMD.png")
plt.show()'''
#?=======================================>







 #todo test------
'''#?(曾用)模态和时间序列------------------------------------------------------------->
#绘图------------------------------------
#?   (一些注释参照 PDO_hindcast.py)
#(在定义北太平洋区域时，经纬度+数据似乎就已经可以定义出该区域了) #?  latest, 2025.03.03
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']         # display Chinese labels normally: SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False                     # display negative sign normally
cmap1=cmaps.BlueDarkRed18                                        # BlueDarkRed18 temp_diff_18lev
list_cmap1=cmap1(np.linspace(0,1,21))
cp=ListedColormap(list_cmap1,name='cp')

fig=plt.figure(figsize=(10,12),dpi=100)  # x*y
proj=ccrs.PlateCarree(central_longitude=180)                         
ax=fig.add_subplot(2,1,1,projection=proj)                               
contourf_plot=ax.contourf(lon, lat, EOF[:,:] , cmap=cp, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
#ax.set_extent([-180,180,-90,90],crs=ccrs.PlateCarree() )         
ax.coastlines(resolution="50m", linewidth=0.8)     
ax.set_title('spatial mode',size=15, pad=10)
#ax.set_xlabel('longitude',size=12)
#ax.set_ylabel('latitude',size=12)
ax.tick_params(size=3, labelsize=12)
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_xticks([180,150,120,-120,-150,-180],crs=ccrs.PlateCarree())      
ax.set_yticks( np.arange(20,71,10), crs=ccrs.PlateCarree())  
#ax.set_aspect(1.5)    # y/x, 1：应该是自适应，越大越厚

norm =mpl.colors.Normalize(vmin=-1, vmax=1)
colorbar = plt.colorbar(contourf_plot, norm=norm, orientation='vertical', extend='both', shrink=1, pad=0.03)
colorbar.set_label('', fontsize=12)
colorbar.set_ticks(np.linspace(-1, 1, 11))  # 设置刻度位置
colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(-1,1,11)],size=12)  # 设置刻度标签

ax2=fig.add_subplot(2,1,2) 
x=np.arange(0,6000)
x1=np.arange(0,5940)
y=PC  # [:,0]
y1=np.empty((5940))
for i in np.arange(30, len(y)-30):
    y1[i-30]=np.mean(y[i-30: i+30])
ax2.plot(x, y, c='r', lw=0.3)
ax2.plot(x1, y1, c='k', lw=1.2)
ax2.set_title('time series', size=15, pad=10)
ax2.set_xticks(np.arange(0,6001,600))
ax2.set_xticklabels(np.arange(0,501,50))
ax2.set_xlabel('year',size=15)
ax2.tick_params(size=3, labelsize=15)
#ax2.set_aspect(0.8)

#Subgraphs position and title-------------------------------------
plt.subplots_adjust(left=0.070,
                    bottom=0.066,
                    right=0.950,
                    top=0.890,
                    wspace=0.18,                 # horizontal distance between subgraphs
                    hspace=0.300                # vertical distance between subgraphs
                )
#plt.tight_layout()
plt.suptitle('PDO of piControl for NorCPM1', fontsize=18, x=0.5, y=0.98)   # x,y=0.5,0.98 (default)
#plt.savefig("D:/decadal prediction/results/piControl/PDO of piControl for NorCPM1.png")
plt.show()
'''#?=========================================================>

#? koe和PDO的相关性
'''# koe area -----
koe_tos=ds.tos[:,120:140,145:210]
koe_tos1=np.array(koe_tos.data).reshape(500,12,20,65)
koe_ac=koe_tos1.mean(axis=0)
koe_tosa=koe_tos1-koe_ac

koe_index=np.nanmean(koe_tosa.reshape(6000,20,65),axis=(1,2))

PC=np.loadtxt("D:/decadal prediction/results/piControl/pdoindex_NorCPM1.txt")
EOF=np.loadtxt("D:/decadal prediction/results/piControl/pdopattern_NorCPM1.txt")
print(EOF.shape)

fig=plt.figure(figsize=(10,8),dpi=100) #koe+PDO
ax=fig.add_subplot()
x=range(0,6000)
y=PC#[:,0]
x1=range(0,5940)
y1=np.empty((5940))
koe1=np.empty((5940))
for i in np.arange(30, len(y)-30):
    y1[i-30]=np.mean(y[i-30: i+30])
    koe1[i-30]=np.mean(koe_index[i-30: i+30])
r,p=pearsonr(y1,koe1)

#ax.plot(x,y,c='r',lw=0.3)
ax.plot(x1,y1,c='r',lw=1,label='PDO index')
#ax.plot(x,koe_index,c='b',lw=0.3)
ax.plot(x1,koe1,c='b',lw=1,label='koe index')
ax.annotate(f'r:{r:.2f}', xy=(4500,1.6), c='k', fontsize=20)


plt.xticks(ticks=np.arange(0,6001,600), labels=np.arange(0,501,50), size=20)  #labels=np.arange(0,501,50),
plt.yticks(ticks=np.arange(-2.0,2.5,0.5), size=20)
plt.xlabel('year',fontsize=20)
plt.legend(loc='upper right',fontsize=16)

plt.show()
'''

''' #todo 但已改进，
# remove the global-mean SST anomaly------------------
sst_g=tos.reshape((500,12,180,360))        # global
sst_gac=np.nanmean(sst_g, axis=0)         # global annual cycle
sst_gaca=np.empty((500,12,180,360))      # global annual cycle anomaly
for j in np.arange(0,12):
    for k in np.arange(0,180):
        for l in np.arange(0,360):
            sst_gaca[:,j,k,l]=sst_g[:,j,k,l]-sst_gac[j,k,l]

# remove the climatological annual cycle------------------------
sst_nps=tos_np.reshape((500,12,50,150))     # np sst 
sst_ac=np.nanmean(sst_nps,axis=0)             # annual cycle

sst2=np.empty((6000,50,150))
for i in np.arange(6000):
    for j in np.arange(0,50):
        for k in np.arange(0,150):
            sst_gma=np.nanmean(sst_gaca[i,:,:])     #global-mean anomaly
            sst2[i,j,k]=tos_np[i,j,k] - sst_gma            

sst3=np.empty((500,12,50,150))
for j in np.arange(0,12):
    for k in np.arange(0,50):
        for l in np.arange(0,150):
            sst3[:,j,k,l]=sst2.reshape((500,12,50,150))[:,j,k,l]-sst_ac[j,k,l]
sst=sst3.reshape((120,50,150))
'''
'''
#todo improved =====
# remove the global-mean SST anomaly------------------
sst_g=tos.reshape((500,12,180,360))        # global
sst_gac=np.nanmean(sst_g, axis=0)         # global annual cycle
sst_gaca=np.empty((500,12,180,360))      # global annual cycle anomaly

# remove the climatological annual cycle------------------------
sst_nps=tos_np.reshape((500,12,50,150))     # np sst 
sst_ac=np.nanmean(sst_nps,axis=0)             # annual cycle

sst2=np.empty((500,12,50,150))
sst3=np.empty((500,12,50,150))
for i in np.arange(0,500):
    for j in np.arange(0,12):

        #piControl 不需要去除全球变暖趋势
        
        #sst_gaca[i,j,:,:]=sst_g[i,j,:,:]-sst_gac[j,:,:]
        #sst_gma=np.nanmean(sst_gaca[i,j,:,:])     # global-mean anomaly
        #sst2[i,j,:,:]=tos_np[i,j,:,:] - sst_gma            # subtract a sequence
        #sst3[i,j,:,:]=sst2[i,j,:,:]-sst_ac[j,:,:]                # subtract a matrix
        
        sst3[i,j,:,:]=tos_np[i,j,:,:]-sst_ac[j,:,:]

sst=sst3.reshape((6000,50,150))

# EOF test---------------------------------------------------------------------------------
lat0=lat.data[:,0]
coslat=np.cos(np.deg2rad(lat0))
weight=np.sqrt(coslat)[...,np.newaxis]                  # latitude weight

solver = Eof(sst, weights=weight)                        # Create EOF solver
EOF= solver.eofsAsCorrelation(neofs=3)             # spatial modes
PC = solver.pcs(npcs=3, pcscaling=1)                 # time series; pcscaling: Scaling of time series -- choose 0,1,2
VAR=solver.varianceFraction(neigs=3)
EOF[0,:,70]=EOF[0,:,71]                                        #循环补全，相同的效果
'''



#todo: significant test ------------------
'''


# 执行显著性检验（实际应用建议n_surrogates≥1000）
significance_level = monte_carlo_significance_test(y, n_surrogates=100)
#? 显著性检验的是单个imf还是整个pdo序列 ？
'''

'''
# ========================
# 3. 改进的显著性检验（分IMF检验）
# ========================
def enhanced_monte_carlo_test(original_signal, n_imfs, n_surrogates=100, confidence=95):
    # 拟合AR1模型参数
    model = AutoReg(original_signal, lags=1)
    model_fit = model.fit()
    ar_coef = model_fit.params[1]
    residuals_std = np.std(model_fit.resid)
    
    # 初始化存储
    surr_periods = [[] for _ in range(n_imfs)]
    
    # 生成替代数据
    for _ in tqdm(range(n_surrogates)):
        # 生成AR1替代数据
        noise = np.random.normal(0, residuals_std, len(original_signal))
        surr = np.zeros_like(original_signal)   #长度与输入数组相同的全为0的数组
        surr[0] = original_signal[0]
        for t in range(1, len(surr)):
            surr[t] = ar_coef * surr[t-1] + noise[t]
        
        # EEMD分解
        eemd_50=EEMD(trials=50)
        surr_imfs = eemd_50.eemd(surr)
        #surr_imfs = perform_eemd(surr)
        
        # 填充周期数据
        for i in range(min(len(surr_imfs), n_imfs)):   #比较两者都有的周期
            period = calculate_imf_period(surr_imfs[i])
            if not np.isnan(period):
                surr_periods[i].append(period)   #每一个数据的第i个周期，往对应的 i 个空列表里面，依次添加n_surrogates次第i个周期
    
    # 计算各IMF显著性阈值，即排序
    sig_levels = []
    for i in range(n_imfs):
        if len(surr_periods[i]) >= 10:  # 最小---样本---量要求
            sig = np.percentile(surr_periods[i], confidence)
        else:
            sig = np.nan
        sig_levels.append(sig)
    
    return sig_levels

# 执行检验
significance_levels = enhanced_monte_carlo_test(y, n_imfs=n_imfs, n_surrogates=1000, confidence=95)

print(significance_levels)
'''

