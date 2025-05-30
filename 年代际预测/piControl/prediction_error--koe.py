
#? PDO事件
#? PDO"指数"用来挑选事件，观测就有了，预测（+-20）也就有了；挑选具有季节预报障碍的预测(600中的哪一个，gr)
#? 挑选SPB对应的各个"变量"的预测误差（观测+预测），高度相关的（EOF），初始+演变
# 显著性检验（95%）并掩膜
#? 指数-->变量
#index_maxerror、spb、apb
#(挑选指数和挑选变量思路大致一样)

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



nmax=15  #挑选15个事件
def find_extrema_indices(data):
    """
    找到原序列中的极值点（导数近似为零的位置）的索引和值。
    - 极大值：导数从正变负
    - 极小值：导数从负变正
    - 极值在7，8，9月份
    """
    n = len(data)
    #if n < 3:
    #    return []
    
    extrema = []
    diff = [data[i+1] - data[i] for i in range(n-1)]
    
    for i in range(1, len(diff)):
        prev_diff = diff[i-1]
        curr_diff = diff[i]
        
        # 极大值：导数从正变负
        if prev_diff > 0 and curr_diff < 0:
            #extrema.append((data[i], i+240))  # 原序列索引为 i; 240: 20year*12
            month_index = i % 12  # 获取该点的月份索引
            if month_index in [5,6,7]:  # 6,7,8月份对应的索引（0表示1月）
                extrema.append((data[i], i + 240))
        # 极小值：导数从负变正
        elif prev_diff < 0 and curr_diff > 0:
            #extrema.append((data[i], i+240))
            month_index = i % 12
            if month_index in [5,6,7]:
                extrema.append((data[i], i + 240))
    
    return extrema

def filter_adjacent_points(sorted_extrema, min_gap=36):
    """
    动态筛选相邻 min_gap 个点内的极值点，确保每个区域内只保留一个值。
    - sorted_extrema: 按值排序后的极值点列表，格式为 [(value, index)]
    - min_gap: 相邻点的最小间隔
    """
    selected = []
    blocked_indices = set()  # 记录已排除的索引范围
    
    for value, idx in sorted_extrema:
        # 检查当前索引是否在已排除的范围内
        is_blocked = any(abs(idx - blocked_idx) <= min_gap for blocked_idx in blocked_indices)
        if not is_blocked:
            selected.append((value, idx))
            # 记录当前索引的排除范围：idx ± min_gap
            blocked_indices.add(idx)
            if len(selected) >= nmax:  # 最多选n个
                break
    return selected

def get_top_bottom_extrema_with_gap(data, k=nmax, min_gap=36):
    """
    返回极值点中最大的 k 个和最小的 k 个值及其原序列索引，并确保相邻点间隔至少 min_gap。
    """
    extrema = find_extrema_indices(data)  #-----
    if not extrema:
        return [], [], [], []
    
    # 按值排序极值点（从大到小）
    sorted_desc = sorted(extrema, key=lambda x: -x[0])
    # 筛选最大的 k 个值，并排除相邻点
    top_extrema = filter_adjacent_points(sorted_desc, min_gap)  #-----
    top_values = [x[0] for x in top_extrema]
    top_indices = [x[1] for x in top_extrema]
    
    # 按值排序极值点（从小到大）
    sorted_asc = sorted(extrema, key=lambda x: x[0])
    # 筛选最小的 k 个值，并排除相邻点
    bottom_extrema = filter_adjacent_points(sorted_asc, min_gap)
    bottom_values = [x[0] for x in bottom_extrema]
    bottom_indices = [x[1] for x in bottom_extrema]
    return top_values, top_indices, bottom_values, bottom_indices



model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
for i in np.arange(0,6):   #!  model*6
    #? 读取数据
    mdl=model[i]
    #?  (I) 指数： 先挑选事件 -------------------------------------------------------------------------------------->
    #PC=np.loadtxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_PDOindex.txt")
    koe_idx=np.loadtxt(f"D:/decadal prediction/results/piControl/{mdl}/koeindex.txt")
    koe_idx=koe_idx/koe_idx.std()
    x=np.arange(0,6000)
    x1=np.arange(0,5940)
    y=koe_idx  # [:,0]
    y1=np.empty((5940))
    for i in np.arange(30, len(y)-30):
        y1[i-30]=np.mean(y[i-30: i+30])  #5 year

    #nmax=15   #? select PDO index maximun value----->

    top_values, top_indices, bottom_values, bottom_indices=get_top_bottom_extrema_with_gap(koe_idx[240:5760],min_gap=36)
    #print(top_values, top_indices, bottom_values, bottom_indices) #?----- PC\y1
    #? =====>36，60，120


    top_indices=np.array(top_indices)
    bottom_indices=np.array(bottom_indices)
    #from collections import Counter
    tquotient,tremainder,bquotient,bremainder=(np.empty(nmax) for _ in np.arange(4))
    for i in np.arange(0,nmax):
        tquotient[i], tremainder[i] = divmod(top_indices[i], 12)       #? 年份+月份
        bquotient[i], bremainder[i] = divmod(bottom_indices[i], 12)
    print('tquotient:',tquotient)  # year
    print('tremainder:',tremainder)

    #tmonth=Counter(tremainder+1)
    #bmonth=Counter(bremainder+1)
    #sorted_tmonth = sorted(tmonth.items())
    #sorted_bmonth = sorted(bmonth.items())
    #print(sorted_tmonth)
    #print(sorted_bmonth)
    #print(bquotient)
    #print(bremainder)

    #? koe events----->
    std=np.std(y1,ddof=1)

    fig=plt.figure(figsize=(16,8),dpi=100)
    ax=fig.add_subplot()
    #ax.plot(x1, y1, c='k', ls='-', lw=2, label='PDOindex_5')
    ax.plot(x,y, c='k', ls='-', lw=0.5, label='koe_index')
    ax.scatter(top_indices,top_values,c='r', marker='o',s=83,label='maximum value')  #240: 20year*12
    ax.scatter(bottom_indices,bottom_values,c='b', marker='o',s=100,label='minimum value')
    #ax.axhline(y=1.5*std,lw=1.2,ls=':',c='r',label='std')
    #ax.axhline(y=-1.5*std,lw=1.2,ls=':',c='r')

    plt.xticks(ticks=np.arange(0,6001,600), labels=np.arange(0,501,50), size=20)  #labels=np.arange(0,501,50),
    plt.yticks(ticks=np.arange(-4.0,6.1), size=20)
    plt.xlabel('year',fontsize=20)

    plt.subplots_adjust(left=0.076,
                        bottom=0.11,
                        right=0.960,
                        top=0.900,
                        wspace=0.2,      #子图间水平距离
                        hspace=0.2     #子图间垂直距离
                    )

    plt.legend(loc='upper right',fontsize=16)#, bbox_to_anchor=(0.80,1))   
    #NorCPM1_PDOevents
    plt.suptitle(f'{mdl}', fontsize=25, x=0.5, y=0.965)   # x,y=0.5,0.98 (default)
    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/koe_events.png')
    #plt.show()
    #?=====>


    #?  事件是使用海表温度定义的，tos
    files=f'F:/data/piControl/{mdl}/tos*.nc'
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


    koe=ds.tos.isel(time=slice(0,6000),
                            y=slice(120,140),
                            x=slice(145,210)).data.compute()
    #print(koe)
    ac=np.nanmean(koe.reshape((500,12,20,65)),axis=(0,2,3))
    year = tquotient     #[(tquotient > 20) & (tquotient < 480)]
    #print('year=',len(year))
    prede=[]
    obs_ens=[]  #所有观测（事件）--指数
    pred_ens=[] #所有预测
    for i in np.arange(0,nmax):
        yr=int(year[i])
        #obs=PC[yr*12:(yr+1)*12]
        obs=koe[yr*12:(yr+1)*12,:,:]
        obs_ens.append(koe_idx[yr*12:(yr+1)*12])
        #print(obs_ens)

        
        for j in np.concatenate([np.arange(-20,0),np.arange(1,21)]): #?-----
            #print(j)
            #pred=PC[(yr+j)*12:(yr+j+1)*12]
            pred=koe[(yr+j)*12:(yr+j+1)*12,:,:]
            pred_ens.append(koe_idx[(yr+j)*12:(yr+j+1)*12])

            #prede.append(np.nanmean(abs(pred-obs)),axis=(1,2)))        #todo-----
            prede.append(np.sqrt(np.nanmean((pred-obs)**2,axis=(1,2))))

    obs_ens=np.array(obs_ens).reshape((nmax,12))
    pred_ens=np.array(pred_ens).reshape((nmax*40,12))
    obs_ens_mean=np.array(obs_ens).reshape((nmax,12)).mean(axis=0)
    pred_ens_mean=np.array(pred_ens).reshape((nmax*40,12)).mean(axis=0)
    #print(obs_ens)

    # 2.包络线-----
    prede=np.array(prede).reshape(nmax*40,12)
    prede_max=np.max(prede,axis=0)
    prede_min=np.min(prede,axis=0)
    prede_mean=np.mean(prede, axis=0)
    print('prede_mean.shape',prede_mean.shape)

    lprede=prede.reshape(nmax,40,12)
    #lpb = np.argpartition([-lprede[i,:,11] for i in np.arange(0,15)], 5)[:5]   #?----- larger pb 前5个
    lpb=[]
    for i in np.arange(0,nmax):
        lpbi=np.argpartition(-lprede[i,:,11], 5)[:5]
        lpbi=lpbi+i*40   #已修改
        lpb.append(lpbi)
    lpb=np.array(lpb).reshape((nmax*5))   # nmax*5
    #np.savetxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_lpb.txt", lpb, fmt='%1.8f') #todo-----
    print(lpb)

    #3. growth rates of prediction error (nmax*40,11)
    growth_rates = (prede[:, 1:] - prede[:, :-1]) / 1  #?-----

    #4. 统计各月成为年度最大增长率的次数 ====
    #max:
    max_month_indices = np.argmax(growth_rates, axis=1)  # 每年最大增长率对应的月份索引（0~10代表1~11月）
    month_counts = np.bincount(max_month_indices, minlength=11)  # 统计次数
    #max2:
    sorted_indices_desc = np.argsort(-growth_rates, axis=1)  # 降序排列的索引
    top2_indices = sorted_indices_desc[:, :3]  # 每行前两个最大的月份
    flattened_top2 = top2_indices.flatten()    # 展平为一维数组
    month_counts_top2 = np.bincount(flattened_top2, minlength=11)
    #print('month_counts:',month_counts)
    #print(month_counts_top2)

    #2_续_. 计算各季节gr------------------
    season_gr=np.empty((nmax*40,4))   #season gr: growth rate
    season_gr[:,0] = (prede[:, 3]-prede[:,0])/3
    season_gr[:,1] = (prede[:, 6]-prede[:,3])/3 #春4、5、6的增长率
    season_gr[:,2]= (prede[:, 9]-prede[:,6])/3
    season_gr[:,3] = (prede[:, 11]-prede[:,9])/2 #秋10,11

    #? 秋季-----筛选
    '''
    aut_column =season_gr[:, 3]   #?-----
    row_max = np.max(season_gr, axis=1)
    mask = (aut_column == row_max)
    apb = np.where(mask)[0]   #Autumn predictability barrier
    #np.savetxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_apb.txt", apb, fmt='%1.8f') #todo-----
    prede_aut=np.array([prede[i,:] for i in apb])
    #print(prede_aut.shape)  (198,12)
    prede_aut_max=np.max(prede_aut,axis=0)
    prede_aut_min=np.min(prede_aut,axis=0)
    prede_aut_mean=np.mean(prede_aut,axis=0)
    '''

    #? 春季-----
    spr_column =season_gr[:, 1]  #?-----
    row_max = np.max(season_gr, axis=1)
    mask = (spr_column == row_max)
    spb = np.where(mask)[0]   #Spring predictability barrier
    np.savetxt(f"D:/decadal prediction/results/piControl/{mdl}/spb.txt", spb, fmt='%1.0f') #todo-----
    print(spb.size)
    prede_spr=np.array([prede[i,:] for i in spb])
    #print(prede_aut.shape)  (198,12)
    prede_spr_max=np.max(prede_spr,axis=0)
    prede_spr_min=np.min(prede_spr,axis=0)
    prede_spr_mean=np.mean(prede_spr,axis=0)



    #? plot: error growth--------------------------------
    fig=plt.figure(figsize=(10,8),dpi=100)
    cmap1=cmaps.BlueDarkRed18                                       # BlueDarkRed18 temp_diff_18lev posneg_1 
    list_cmap1=cmap1(np.linspace(0,1,10))
    cp=ListedColormap(list_cmap1,name='cp')

    ax1=fig.add_subplot(221)   #predictiob error
    #todo 逐月的误差增长率，逐季的误差增长率, done.
    x1=np.arange(0,12)
    for i in np.arange(0,nmax*40):
        ax1.plot(x1,prede[i,:])
        ax1.set_xlabel('month',size=16)
        ax1.set_ylabel('prediction error',size=16)
        plt.xticks(ticks=np.arange(0,12),labels=np.arange(1,13))
        #plt.xticks(ticks=([150,300,450]), size=12)
        ax1.tick_params(size=6, labelsize=16)
        plt.yticks(ticks=np.arange(0,8))

    ax2=fig.add_subplot(222)   #prediction error(spr、apr)
    x2=np.arange(0,12)
    ax2.plot(x2,prede_mean,c='b',lw=1.5,label='All', zorder=4)
    ax2.plot(x2,prede_spr_mean,c='r',lw=1.5,label='Spring', zorder=4) #g
    #ax2.plot(x2,prede_aut_mean,c='r',lw=1.5,label='Autumn', zorder=4)
    ax2.fill_between(x2, prede_max, prede_min, color='LightSkyBlue', alpha=1, zorder=1)
    ax2.fill_between(x2, prede_spr_max, prede_spr_min, color='orangered', alpha=0.6, zorder=1) #LimeGreen
    #ax2.fill_between(x2, prede_aut_max, prede_aut_min, color='Salmon', alpha=0.8, zorder=3)
    ax2.set_xlabel('month',size=16)
    ax2.set_ylabel('prediction error',size=16)
    plt.xticks(ticks=np.arange(0,12),labels=np.arange(1,13))
    plt.yticks(ticks=np.arange(0,7))
    #plt.xticks(ticks=([150,300,450]), size=12)
    ax2.tick_params(size=6, labelsize=16)
    ax2.legend(fontsize=12)

    ax3=fig.add_subplot(223)  # growth rate contourf
    x3=np.arange(0,nmax*40)
    y3=np.arange(0,11)
    X3,Y3=np.meshgrid(x3,y3) #?-----值得注意
    contourf_plot=ax3.contourf(X3,Y3,growth_rates.T,cmap=cp, levels=np.linspace(-1,1,11), extend="both")
    ax3.set_xlabel('number',size=16)
    ax3.set_ylabel('month',size=16)
    ax3.set_xticks(ticks=np.arange(0,nmax*40+1,100))
    ax3.set_yticks(ticks=np.arange(0,11,3),labels=[1,4,7,10],size=16)
    ax3.tick_params(size=6,labelsize=16)
    ax3.set_title('growth rates',size=16, pad=8)

    norm =mpl.colors.Normalize(vmin=-1, vmax=1)
    colorbar = plt.colorbar(contourf_plot, norm=norm, orientation='vertical', extend='both', shrink=1, pad=0.05)
    colorbar.set_label('', fontsize=12)
    colorbar.set_ticks(np.linspace(-1, 1, 11))  # 设置刻度位置
    colorbar.set_ticklabels([f"{tick:.1f}" for tick in np.linspace(-1,1,11)],size=13)  # 设置刻度标签

    ax4=fig.add_subplot(224)   # growth rate number
    x4=np.arange(0,11)
    ax4.bar(x4, month_counts, width=0.6)
    plt.xticks(ticks=np.arange(0, 11), labels=np.arange(1,12))
    ax4.set_xlabel('month',size=16)
    ax4.set_ylabel('number',size=16)
    ax4.tick_params(size=6,labelsize=16)
    #plt.yticks(ticks=np.arange(0,201,50))

    plt.subplots_adjust(left=0.080,
                        bottom=0.098,
                        right=0.980,
                        top=0.915,
                        wspace=0.2,      #子图间水平距离
                        hspace=0.421     #子图间垂直距离
                    )
    #CanESM2_index_prediction_error
    plt.suptitle(f'{mdl}', fontsize=25, x=0.5, y=0.983)   # x,y=0.5,0.98 (default)
    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/koe_idx_errors.png')
    print('--------------------------.')


    #观测和预测，及其集合平均
    '''fig=plt.figure(figsize=(10,8),dpi=100)
    ax=fig.add_subplot()
    for i in np.arange(0,nmax):
        ax.plot(np.arange(1,13),obs_ens[i,:],c='r',lw=0.5,ls='--')
    ax.plot(np.arange(1,13),obs_ens_mean,c='r',lw=5,ls='-',label='observation')
    for i in np.arange(0,nmax*40):
        ax.plot(np.arange(1,13),pred_ens[i,:],c='b',lw=0.5,ls='--')
    ax.plot(np.arange(1,13),pred_ens_mean,c='b',lw=5,ls='-',label='prediction')
    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/obs+pred.png')'''

    #plt.show()
    #? ============================================================>



    #--->spb 传递

    #todo (III) 前面是指数；后面开始变量
    '''
    # 有spb、apb的预测的初始误差结构（对其相应变量的初始误差进行EOF，并挑选高度相关的进行合成）；
    # 15*40=600；divmod（spb[0],40(-20)）
    # 哪个事件的哪次预报的，哪个变量
    '''

    #? 相关变量------------------------------------------------------------------->
    # variables：tos、thetao、u、v、z、mld  #! 变量*4
    #读取数据---------------------------
    #files = 'D:/decadal prediction/data/piControl/NorCPM1/tos*.nc'   
    variables=['tos','ua','va','thetao','thetao','thetao']   #u[0]、v[0]、tos放一个循环里；thetao：5，9，13level
    for j in np.arange(3,5):    #!-----
        variable=variables[j]
        files=f'F:/data/piControl/{mdl}/{variable}*.nc'  #?---
        combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
        if j==1 or j==2:
            combined_ds=combined_ds.isel(plev=0)  #表面风u0、v0
        if j==3:
            combined_ds=combined_ds.sel(lev=50, method='nearest')  #50m附近层的thetao
        if j==4:
            combined_ds=combined_ds.sel(lev=100, method='nearest')
        if j==5:
            combined_ds=combined_ds.sel(lev=130, method='nearest')
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

        # 筛选: spb、apb、lpb ----------
        #? (可删除)  spb=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/spb.txt')
        #apb=np.loadtxt('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_apb.txt')
        #lpb=np.loadtxt('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_lpb.txt')
        spby,spbx=divmod(spb,40)   #600=15*40-->spb.shape(year+x_pred)； float
        spby = spby.astype(int)
        spbx = spbx.astype(int)
        #print(spbx)
        #在事件前后筛选预测
        spbx=spbx-20
        spbx = [x + 1 if x >= 0 else x for x in spbx]   #?-----
        #print(spbx)

        # 初始误差、变量区域(南纬20度以北的太平洋)   
        #? 季节预报障碍的基础上，进一步分析到高度相关的初始误差----------------------------
        #spby\spbx长度是一样的；y=spby[i];x=spbx[i];y是某个事件，x是波动年份
        variable_p=ds[variable][:,70:160,120:280]   # sst pacific_related: 20S-70N, 120E-80W; PDO:20N-70N, 120E-100W
        lon=variable_p.lon     #?-----
        lat=variable_p.lat

        #区分不同层的thetao；variable在此之后的作用仅剩保存的作用
        if j==3:
            variable='thetao-50'
        if j==4:
            variable='thetao-100'
        if j==5:
            variable='thetao-130'

        #print(lon,lat)

        lenspb=len(spb)
        #print(lenspb)
        #!---加循环--月份的误差
        months=[0,2,5,8,11]   #1,3,6,9,12月

        #if variable=='ua' or variable=='va':
        #    months=[-1,1,4,7,10]     #? u、v提前一个月，风和海温的超前滞后关系

        for k in np.arange(0,1):  #!-----   
            month=months[k]
            errors=np.empty((lenspb,90,160))
            #ij=np.empty((lenspb,2))
            all_times = []
            for i in np.arange(0,lenspb):
                events = spby[i]   #todo-----5
                yrs = spbx[i] 
                t1 = int(tquotient[events] * 12+month)   #?   -7、-4、-1、0、2、5、8、11、14、17----- #观测
                t2 = int((tquotient[events] + yrs) * 12+month)                                                                      #预测
                all_times.extend([t1, t2])  # 保留重复的时间点
            preloaded_data = variable_p.isel(time=all_times).load()   #一次性读取所需数据点，重复O/I很占用...线程？

            #month=month+1  #?-----

            for i in np.arange(0,lenspb):      #todo -----
                idx1 = 2 * i      
                idx2 = 2 * i + 1  
                
                observation = preloaded_data.data[idx1]   #观测
                prediction = preloaded_data.data[idx2]      #预测
                
                errors[i] = prediction - observation
                #print(i)
            print(errors.shape)

            ''' #? ever
            for i in np.arange(0,lenapb):
                events=apby[i]   #todo-----5
                yrs=apbx[i]  
                observation=variable_p[int(tquotient[events]*12),:,:]             #todo 初始误差   #演变 :(tquotient[i]+1)*12
                prediction=variable_p[int((tquotient[events]+yrs)*12),:,:]         #初始误差   #演变:(tquotient[i]+j+1)*12
                error=(prediction-observation).data                                            #有NaN值
                errors[i] = error       #?
                #ij[i,0]=events
                #ij[i,1]=yrs
                #ij.append(events)   
                #ij.append(yrs)
                print(i)
            print(errors.shape)
            '''

            #? EOF test---------------------------------------------------------------------------------
            lat0=lat.data
            coslat=np.cos(np.deg2rad(lat0))
            weight=np.sqrt(coslat)[...,np.newaxis]                  # latitude weight
            print(weight.shape)   #(90,1) √
            solver = Eof(errors, weights=weight)                   #? Create EOF solver
            EOF= solver.eofsAsCorrelation(neofs=3)             # spatial modes
            PC = solver.pcs(npcs=3, pcscaling=1)                 # time series; pcscaling: Scaling of time series -- choose 0,1,2
            VAR=solver.varianceFraction(neigs=3)

            #pc1=PC[:,0]
            #pc2=PC[:,1]
            #pc3=PC[:,2]
            print(VAR)
            #np.savetxt('pc1.txt',pc1, fmt='%1.8f')

            if month==0:

                pc1=PC[:,0]
                pc2=PC[:,1]
                pc3=PC[:,2]
                pcx=[pc1,pc2,pc3]
                for l in np.arange(0,3):

                    arr=pcx[i]   #?-----

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
            #pcx=[pc1,pc2,pc3]
            composition=np.empty((6,90,160))
            ppn=np.empty((6,90,160))
            for i in np.arange(0,3):
                #arr=pcx[i]   #?-----

                #positive_values = arr[arr > 0]
                #positive_mean = positive_values.mean()
                #negative_mean = negative_values.mean()
                #positive_indices = np.where(arr > positive_mean)[0]
                #negative_indices = np.where(arr < negative_mean)[0]
                #print(positive_indices.size)
                #print(negative_indices.size)
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
            #print(positive_indices,positive_mean,positive_values)
            #print(np.max(eofs[0,:,:]),np.min(eofs[0,:,:]))
            #print(np.max(eofs[1,:,:]),np.min(eofs[1,:,:]))
            #print(np.max(eofs[2,:,:]),np.min(eofs[2,:,:]))

            #? 6个模式，哪种情况下（spb、apb、lpb）哪个变量（）的哪个月的---误差
            with open(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_errors_{month+1}.txt', 'w') as f:  #i,j,k: 模式、变量、月份
                # 遍历数组的每个切片
                for two_d_slice in errors:      #?--- 误差
                    np.savetxt(f, two_d_slice, fmt='%1.4f', delimiter=' ')
                    f.write('\n')


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
                
                '''  #ever
                # 遍历数组的每个切片
                for two_d_slice in errors:      #?--- 误差
                    # 将二维切片转换为字符串，元素之间用空格分隔
                    for row in two_d_slice:
                        f.write(' '.join(map(str, row)) + '\n')
                    # 每个切片之间添加一个空行
                    f.write('\n')
                '''

            
            print(np.nanmax(composition))   #方便设置colorbar

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

    








'''#? 曾临时借用------------------
eofs=np.loadtxt('D:/decadal prediction/results/piControl/NorCPM1_tos_initial_error_eofs.txt')
eofs=eofs.reshape((6,90,160))
files = 'D:/decadal prediction/data/piControl/NorCPM1/Tos*.nc'  # 替换为你的文件夹路径

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
sstp=ds.tos[:,70:160,120:280]   #sst pacific_related: 20S-70N, 120E-80W; PDO:20N-70N, 120E-100W
lon=sstp.lon
lat=sstp.lat
'''

