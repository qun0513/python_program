
#?decadal prediction：10periods
#观测+预测
#特定的预测
#高度相关的初始误差


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
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import pearsonr
import heapq
import math
from scipy.stats import ttest_1samp





def find_prediction(target_number):
    result = []
    for num in range(0, 490):
        if num != target_number and abs(num - target_number) % 10 == 0:
            result.append(num)
    return result

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
    x2=np.arange(0,5880)
    y=koe_idx  # [:,0]
    y1=np.empty((5940))    #-->10年周期
    y2=np.empty((5880))    #-->20年周期
    for i in np.arange(30, len(y)-30):
        y1[i-30]=np.mean(y[i-30: i+30])  #5 year
    y1=y1/y1.std()
    for i in np.arange(60, len(y)-60):
        y2[i-60]=np.mean(y[i-60: i+60])  #10 year
    y2=y2/y2.std()
    #nmax=15   #? select PDO index maximun value----->

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
                extrema.append((data[i], i+240))  # 原序列索引为 i; 240: 20year*12
                #month_index = i % 12  # 获取该点的月份索引
                #if month_index in [5,6,7]:  # 6,7,8月份对应的索引（0表示1月）
                #    extrema.append((data[i], i + 240))
            # 极小值：导数从负变正
            elif prev_diff < 0 and curr_diff > 0:
                extrema.append((data[i], i+240))
                #month_index = i % 12
                #if month_index in [5,6,7]:
                #    extrema.append((data[i], i + 240))
        
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
    top_values, top_indices, bottom_values, bottom_indices=get_top_bottom_extrema_with_gap(y[240:5760],min_gap=36)

    nmax=10  #挑选10个事件
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
                extrema.append((data[i], i))  # 原序列索引为 i; 240: 20year*12
                #month_index = i % 12  # 获取该点的月份索引
                #if month_index in [5,6,7]:  # 6,7,8月份对应的索引（0表示1月）
                #    extrema.append((data[i], i + 240))
            # 极小值：导数从负变正
            elif prev_diff < 0 and curr_diff > 0:
                extrema.append((data[i], i))
                #month_index = i % 12
                #if month_index in [5,6,7]:
                #    extrema.append((data[i], i + 240))
        
        return extrema

    def filter_adjacent_points(sorted_extrema, min_gap=120):
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

    def get_top_bottom_extrema_with_gap(data, k=nmax, min_gap=120):
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
    top_values1, top_indices1, bottom_values1, bottom_indices1=get_top_bottom_extrema_with_gap(y1[:-120],min_gap=108)
    #print(top_values, top_indices, bottom_values, bottom_indices) #?----- PC\y1
    #? =====>36，60，120

    #? 年份+月份
    top_indices1=np.array(top_indices1)
    bottom_indices1=np.array(bottom_indices1)
    #from collections import Counter
    tquotient1,tremainder1,bquotient1,bremainder1=(np.empty(nmax) for _ in np.arange(4))
    for i in np.arange(0,nmax):
        tquotient1[i], tremainder1[i] = divmod(top_indices1[i], 12)       #? 年份+月份
        bquotient1[i], bremainder1[i] = divmod(bottom_indices1[i], 12)
    #print('tquotient:',tquotient1)  # year
    print('tremainder:',tremainder1)

    #? koe events----->
    std=np.std(y1,ddof=1)

    fig=plt.figure(figsize=(16,8),dpi=100)
    ax=fig.add_subplot()
    #ax.plot(x1, y1, c='k', ls='-', lw=2, label='PDOindex_5')
    
    #ax.axhline(y=1.5*std,lw=1.2,ls=':',c='r',label='std')
    #ax.axhline(y=-1.5*std,lw=1.2,ls=':',c='r')

    ax.plot(x,y, c='k', ls='-', lw=0.2, label='koe_index')
    ax.scatter(top_indices,top_values,c='r', marker='o',s=20,label='maximum value')  #240: 20year*12
    ax.scatter(bottom_indices,bottom_values,c='b', marker='o',s=20,label='minimum value')

    ax.plot(x1,y1, c='k', ls='-', lw=1.5, label='koe_index_10')
    ax.scatter(top_indices1,top_values1,c='r', marker='D',s=60,label='maximum value_10')  #240: 20year*12
    ax.scatter(bottom_indices1,bottom_values1,c='b', marker='D',s=60,label='minimum value_10')

    #ax.plot(x2,y2, c='k', ls='-', lw=3, label='koe_index_20')
    #ax.scatter(top_indices2,top_values2,c='r', marker='^',s=100,linewidths=3,label='maximum value_20')  #240: 20year*12
    #ax.scatter(bottom_indices2,bottom_values2,c='b', marker='^',s=100,linewidths=3,label='minimum value_20')

    plt.xticks(ticks=np.arange(0,6001,600), labels=np.arange(0,501,50), size=20)  #labels=np.arange(0,501,50),
    plt.yticks(ticks=np.arange(-4.0,7.1), size=20)
    plt.xlabel('year',fontsize=20)

    plt.subplots_adjust(left=0.076,
                        bottom=0.11,
                        right=0.960,
                        top=0.900,
                        wspace=0.2,      #子图间水平距离
                        hspace=0.2     #子图间垂直距离
                    )

    plt.legend(loc='upper right',fontsize=16,ncol=2)#, bbox_to_anchor=(0.80,1))   
    #NorCPM1_PDOevents
    plt.suptitle(f'{mdl}', fontsize=25, x=0.5, y=0.965)   # x,y=0.5,0.98 (default)
    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/koe_events_10.png')
    #plt.show()
    plt.close()
    #?=====>


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
    #ac=np.nanmean(koe.reshape((500,12,20,65)),axis=(0,2,3))
    year = tquotient1     #[(tquotient > 20) & (tquotient < 480)]
    #print('year=',len(year))
    prede=[]
    yrs=np.empty((nmax,48))
    for i in np.arange(0,nmax):
        yr=int(year[i])
        obs=koe[yr*12:(yr+10)*12,:,:]

        years=find_prediction(yr)   # yrs--48
        yrs[i]=np.array(years)
        for j in years:   #?-----
            pred=koe[(j)*12:(j+10)*12,:,:]

            #prede.append(np.nanmean(abs(pred-obs)),axis=(1,2)))        #todo-----
            prede.append(np.sqrt(np.nanmean((pred-obs)**2,axis=(1,2)))) #    /   np.nanmean(pred-obs,axis=(1,2)) \ np.sqrt(np.nanmean((pred-obs)**2,axis=(1,2)))
            #prede.append(np.nanmean(pred-obs,axis=(1,2)))
    
    prede=np.array(prede).reshape(nmax*48,120) 
    np.savetxt(f"D:/decadal prediction/results/piControl/{mdl}_prede_10.txt",prede)
    
    #nmax=10
    #prede=np.loadtxt(f"D:/decadal prediction/results/piControl/{mdl}_prede_10.txt")
    lprede=np.array(prede).reshape(nmax,48,120)
    #lpb = np.argpartition([-lprede[i,:,11] for i in np.arange(0,15)], 5)[:5]   #?----- larger pb 前5个
    lpb=[]
    for i in np.arange(0,nmax):
        lpbi=np.argpartition((-np.sum(lprede[i,:,53:56],axis=1)-np.sum(lprede[i,:,113:116],axis=1)), 5)[:5]    #(-lprede[i,:,54]-lprede[i,:,114]) \ -np.sum(lprede[i,:,108:120],axis=1)
        lpbi=lpbi+i*48 
        lpb.append(lpbi)
    lpb=np.array(lpb).reshape((nmax*5))   # nmax*5
    #np.savetxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_lpb.txt", lpb, fmt='%1.8f') #todo-----
    print(lpb)
    lp=prede[lpb]
    print(lp.shape)

    growth_rates = (prede[:, 1:] - prede[:, :-1]) / 1
    #gr_year=growth_rates.reshape(480,10,12).mean(axis=2)


    fig=plt.figure(figsize=(10,8),dpi=100)
    cmap1=cmaps.BlueDarkRed18                                       # BlueDarkRed18 temp_diff_18lev posneg_1 
    list_cmap1=cmap1(np.linspace(0,1,10))
    cp=ListedColormap(list_cmap1,name='cp')

    ax1=fig.add_subplot(211)   #predictiob error
    #todo 逐月的误差增长率，逐季的误差增长率, done.
    x1=np.arange(0,120)
    for i in np.arange(0,nmax*48):
        ax1.plot(x1,prede[i,:],c='b')        #all prede
        ax1.set_xlabel('month',size=16)
        ax1.set_ylabel('prediction error',size=16)
        plt.xticks(ticks=np.arange(0,120,12),labels=np.arange(1,121,12))
        #plt.xticks(ticks=([150,300,450]), size=12)
        ax1.tick_params(size=6, labelsize=16)
        #plt.yticks(ticks=np.arange(0,8))
    tks=np.arange(6,126,12)
    for i in np.arange(0,len(lp)):
        ax1.plot(x1,lp[i,:],c='r',lw='1')
        plt.xticks(ticks=np.arange(6,126,12),labels=np.arange(7,127,12))
    for i in np.arange(0,10):
        ax1.axvline(x=tks[i],lw=1.2,ls=':',c='gray')
    
    ax3=fig.add_subplot(212)  # growth rate contourf
    x3=np.arange(0,nmax*48)
    y3=np.arange(0,119)
    X3,Y3=np.meshgrid(x3,y3) #?-----值得注意
    contourf_plot=ax3.contourf(X3,Y3,growth_rates.T,cmap=cp, levels=np.linspace(-1,1,11), extend="both")
    ax3.set_xlabel('number',size=16)
    ax3.set_ylabel('month',size=16)
    ax3.set_xticks(ticks=np.arange(0,nmax*48+1,100))
    plt.yticks(ticks=np.arange(4,124,12),labels=np.arange(5,125,12))
    ax3.tick_params(size=6,labelsize=16)
    ax3.set_title('growth rates',size=16, pad=8)

    plt.suptitle(f"{mdl}",fontsize=20)
    plt.subplots_adjust(left=0.150,
                    bottom=0.098,
                    right=0.950,
                    top=0.915,
                    wspace=0.2,      #子图间水平距离
                    hspace=0.421     #子图间垂直距离
                )
    plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/koeidx_errors10.png')
    #plt.show()
    plt.close()
'''
    variables=['tos','ua','va','thetao','thetao','thetao']   #u[0]、v[0]、tos放一个循环里；thetao：5，9，13level
    for j in np.arange(0,1):    #!-----
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
        lpby,lpbx=divmod(lpb,48)   #600=15*40-->spb.shape(year+x_pred)； float
        lpby = lpby.astype(int)  #第几个事件
        lpbx = lpbx.astype(int)  #的第几个预测
        print(lpby,lpbx)

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

        lenlpb=len(lpb)
        #errors=np.empty((lenlpb,90,160))
        #ij=np.empty((lenspb,2))
        
        phases=[0,6,6+12,6+12*2,6+12*3,6+12*4,6+12*5,6+12*6,6+12*7,6+12*8,6+12*9]   #!看10年周期中哪个年份的误差
        for k in np.arange(0,1):   #len(phases)
            phase=phases[k]
            all_times = []
            errors=np.empty((lenlpb,90,160))
            for i in np.arange(0,lenlpb):
                events = lpby[i]   #todo-----5
                pdt = lpbx[i] 
                t1 = int(tquotient1[events] * 12+phase)   #?   -7、-4、-1、0、2、5、8、11、14、17----- #观测
                t2 = int((yrs[events,pdt] ) * 12+phase)                                                                      #预测
                all_times.extend([t1, t2])  # 保留重复的时间点
            preloaded_data = variable_p.isel(time=all_times).load()   #一次性读取所需数据点，重复O/I很占用...线程？

            #month=month+1  #?-----

            for i in np.arange(0,lenlpb):      #todo -----
                idx1 = 2 * i      
                idx2 = 2 * i + 1  
                
                observation = preloaded_data.data[idx1]   #观测
                prediction = preloaded_data.data[idx2]      #预测
                
                errors[i] = prediction - observation
                #print(i)
            print(errors.shape)


            #? EOF test---------------------------------------------------------------------------------
            lat0=lat.data
            coslat=np.cos(np.deg2rad(lat0))
            weight=np.sqrt(coslat)[...,np.newaxis]                  # latitude weight
            print(weight.shape)   #(90,1) √
            solver = Eof(errors, weights=weight)                   #? Create EOF solver
            EOF= solver.eofsAsCorrelation(neofs=3)             # spatial modes
            PC = solver.pcs(npcs=3, pcscaling=1)                 # time series; pcscaling: Scaling of time series -- choose 0,1,2
            VAR=solver.varianceFraction(neigs=3)

            pc1=PC[:,0]
            pc2=PC[:,1]
            pc3=PC[:,2]
            print(VAR)
            #np.savetxt('pc1.txt',pc1, fmt='%1.8f')

            #? 挑选正（负）高度相关的初始误差，并进行合成-------------
            pcx=[pc1,pc2,pc3]
            composition=np.empty((6,90,160))
            ppn=np.empty((6,90,160))
            for i in np.arange(0,3):
                arr=pcx[i]   #?-----

                positive_values = arr[arr > 0]
                negative_values = arr[arr < 0]
                positive_mean = positive_values.mean()
                negative_mean = negative_values.mean()
                positive_indices = np.where(arr > positive_mean)[0]
                negative_indices = np.where(arr < negative_mean)[0]
                print(positive_indices.size)
                print(negative_indices.size)

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
            with open(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_errors10_{phase+1}.txt', 'w') as f:  #i,j,k: 模式、变量、月份
                # 遍历数组的每个切片
                for two_d_slice in errors:      #?--- 误差
                    np.savetxt(f, two_d_slice, fmt='%1.4f', delimiter=' ')
                    f.write('\n')


            with open(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_composition10_{phase+1}.txt', 'w') as f:
                # 遍历数组的每个切片
                for two_d_slice in composition:      #?--- 合成的误差（高度相关）
                    np.savetxt(f, two_d_slice, fmt='%1.4f', delimiter=' ')
                    f.write('\n')


            with open(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_ppn10_{phase+1}.txt', 'w') as f:
                # 遍历数组的每个切片
                for two_d_slice in ppn:      #?--- 显著性检验
                    np.savetxt(f, two_d_slice, fmt='%1.4f', delimiter=' ')
                    f.write('\n')
                
                #ever
                # 遍历数组的每个切片
                #for two_d_slice in errors:      #?--- 误差
                #    # 将二维切片转换为字符串，元素之间用空格分隔
                #    for row in two_d_slice:
                #        f.write(' '.join(map(str, row)) + '\n')
                #    # 每个切片之间添加一个空行
                #    f.write('\n')


            
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
            plt.suptitle(f'{mdl}_{variable}_errors10_{phase+1}', fontsize=24, x=0.5, y=0.98)# x,y=0.5,0.98(默认)
            plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/{variable}_errors10_{phase+1}.png')
            #plt.show()
            #?==========================================>
        
        #plt.show()#block=False
        #plt.pause(2)  # 显示图像2秒
        plt.close('all')
    phas=[0,6,6+12,6+12*2,6+12*3,6+12*4,6+12*5,6+12*6,6+12*7,6+12*8,6+12*9]
    for kk in np.arange(0,1):
        pha=phas[kk]
        u_comp=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/ua_composition10_{pha+1}.txt')
        u_c=u_comp.reshape((6,90,160))
        v_comp=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/va_composition10_{pha+1}.txt')
        v_c=v_comp.reshape((6,90,160))
        t_comp=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/tos_composition10_{pha+1}.txt')
        t_c=t_comp.reshape((6,90,160))
        t_ppn=np.loadtxt(f'D:/decadal prediction/results/piControl/{mdl}/tos_ppn10_{pha+1}.txt')
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
        plt.suptitle(f'{mdl}_uvt_errors10_{pha+1}', fontsize=24, x=0.5, y=0.98)# x,y=0.5,0.98(默认)
        plt.savefig(f'D:/decadal prediction/results/piControl/{mdl}/uvt_errors10_{pha+1}.png')

'''


