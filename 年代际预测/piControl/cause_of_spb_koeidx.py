

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

import matplotlib.pyplot as plt
import numpy as np


model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
for i in np.arange(1,2):   #  model*6
    mdl=model[i]
    variables=['ua','tos'] #'mld'
    pc=np.empty((2,6000))
    for j in np.arange(0,2):
        variable=variables[j]
        files=f'F:/data/piControl/{mdl}/{variable}*.nc'  #?---
        combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
        #print(combined_ds)

        if j==0 :
            combined_ds=combined_ds.isel(plev=0)  #表面风u0、v0
        
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
        print(ds)

        koe=ds[variable].isel(time=slice(0,6000),
                            y=slice(110,160),
                            x=slice(120,260)).data.compute()
        
        lon=ds[variable].lon.isel(x=slice(120,260)).data
        lat=ds[variable].lat.isel(y=slice(110,160)).data
        #print(lon)
        #print(lat)
        coslat=np.cos(np.deg2rad(lat))
        weight=np.sqrt(coslat)[...,np.newaxis]   #纬度权重
        print(weight)

        solver = Eof(koe, weights=weight)                       # 创建EOF求解器
        EOF= solver.eofsAsCorrelation(neofs=3)             # 计算前几个EOF空间模态
        PC = solver.pcs(npcs=3, pcscaling=1)                 # 计算前三个主成分时间序列; pcscaling:时间序列的缩放形式--可取0,1,2
        VAR=solver.varianceFraction(neigs=3)
        print('VAR:',VAR)
        pc[j]=PC[:,0]

    '''
            cc=np.empty(24)
    ccc=np.empty(25)
    for k in np.arange(1,25):
        if k<12:
            cc[k-1],p=pearsonr(pc[0,:-k],pc[1,k:])
        if k>=12:
            cc[k-1],p=pearsonr(pc[1,:-k+11],pc[0,k-11:])
    
    for k in np.arange(0,12):
        ccc[k]=cc[11-k]
        ccc[13+k]=cc[23-k]
    ccc[12],p=pearsonr(pc[0,:],pc[1,:])
    c,p=pearsonr(pc[0,:],pc[1,:])'''
    
    from statsmodels.tsa.stattools import ccf
    # 将数据转换为 pandas Series
    a = pd.Series(pc[0,:])
    b = pd.Series(pc[1,:])

    # 定义函数计算超前滞后相关
    def calculate_lead_lag_correlation(a, b, max_lag=12):
        # 计算交叉相关函数
        cross_correlation = ccf(a, b, adjusted=False)
        
        # 计算 a 超前 b 的相关性（1-12 个月）
        a_lead_b = cross_correlation[1:max_lag + 1]  # a 超前 b 的相关性
        
        # 计算 b 超前 a 的相关性（1-12 个月）
        b_lead_a = cross_correlation[-max_lag - 1:-1]  # b 超前 a 的相关性
        
        return a_lead_b, b_lead_a

    # 计算超前滞后相关
    max_lag = 12
    a_lead_b, b_lead_a = calculate_lead_lag_correlation(a, b, max_lag)
    print(a_lead_b)
    ab,p=pearsonr(pc[0,:],pc[1,:])
    print(ab)
    ab = np.array([ab])
    cc=np.concatenate([a_lead_b[::-1],ab,b_lead_a])

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(np.arange(-12,13),cc)
    plt.xticks(ticks=np.arange(-12,13))
    plt.show() 




        


            






data = np.random.randn(5,12)

# 绘制箱线图
fig=plt.figure(figsize=(8, 6))
ax=fig.add_subplot()
ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'),
            showmeans=True,meanprops=dict(marker='d')
            )
# 添加标题和标签
ax.tick_params(size=6, labelsize=16)
ax.set_xlabel('Data', size=16)
ax.set_ylabel('Value', size=16)
ax.set_title('Box Plot Example', size=16)

# 显示图形
plt.show()


