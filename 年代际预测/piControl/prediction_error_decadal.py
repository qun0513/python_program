 #? PDO事件
#? PDO"指数"用来挑选事件，观测就有了，预测（+-20）也就有了；挑选具有季节预报障碍的预测(600中的哪一个，gr)
#? 挑选SPB对应的各个"变量"的预测误差（观测+预测），高度相关，初始+演变
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

#?  (I) 指数： 先挑选事件 -------------------------------------------------------------------------------------->
PC=np.loadtxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_PDOindex.txt")

x=np.arange(0,6000)
x1=np.arange(0,5940)
y=PC  # [:,0]
y1=np.empty((5940))
for i in np.arange(30, len(y)-30):
    y1[i-30]=np.mean(y[i-30: i+30])

nmax=10   #? select PDO index maximun value----->
def find_extrema_indices(data):
    """
    找到原序列中的极值点（导数近似为零的位置）的索引和值。
    - 极大值：导数从正变负
    - 极小值：导数从负变正
    """
    n = len(data)
    if n < 3:
        return []
    
    extrema = []
    diff = [data[i+1] - data[i] for i in range(n-1)]
    
    for i in range(1, len(diff)):
        prev_diff = diff[i-1]
        curr_diff = diff[i]
        
        # 极大值：导数从正变负
        if prev_diff > 0 and curr_diff < 0:
            extrema.append((data[i], i+624))  # 原序列索引为 i; 240: 20year*12
        # 极小值：导数从负变正
        elif prev_diff < 0 and curr_diff > 0:
            extrema.append((data[i], i+624))
    
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

top_values, top_indices, bottom_values, bottom_indices=get_top_bottom_extrema_with_gap(y1[624:5376],min_gap=120)
#print(top_values, top_indices, bottom_values, bottom_indices) #?----- PC\y1
#? =====>36，60，120


top_indices=np.array(top_indices)
bottom_indices=np.array(bottom_indices)

tquotient,tremainder,bquotient,bremainder=(np.empty(nmax) for _ in np.arange(4))
for i in np.arange(0,nmax):
    tquotient[i], tremainder[i] = divmod(top_indices[i], 12)       #? 年份+月份
    bquotient[i], bremainder[i] = divmod(bottom_indices[i], 12)
print('tquotient:',tquotient)  # year
print('tremainder:',tremainder)
#print(bquotient)
#print(bremainder)

#? PDO events----->
std=np.std(y1,ddof=1)

fig=plt.figure(figsize=(10,8),dpi=100)
ax=fig.add_subplot()
ax.plot(x1, y1, c='k', ls='-', lw=2, label='PDOindex_5')
#ax.plot(x,y, c='k', ls='-', lw=0.5, label='PDOindex')
ax.scatter(top_indices,top_values,c='r', marker='o',s=83,label='maximum value')  #240: 20year*12
ax.scatter(bottom_indices,bottom_values,c='b', marker='o',s=100,label='minimum value')
#ax.axhline(y=1.5*std,lw=1.2,ls=':',c='r',label='std')
#ax.axhline(y=-1.5*std,lw=1.2,ls=':',c='r')

plt.xticks(ticks=np.arange(0,6001,600), labels=np.arange(0,501,50), size=20)  #labels=np.arange(0,501,50),
plt.yticks(ticks=np.arange(-2.0,2.5,1), size=20)
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
plt.suptitle('NorCPM1_10', fontsize=25, x=0.5, y=0.965)   # x,y=0.5,0.98 (default)
plt.show()
#?=====>



#? (II) PDO index prediction error-------------------------------------------------------------------------------------------->
#? prediction+pb-----
year = tquotient     #[(tquotient > 20) & (tquotient < 480)]
#print('year=',len(year))
prede=[]
for i in np.arange(0,nmax):
    yr=int(year[i])
    obs=PC[(yr-2)*12:(yr)*12]
    
    for j in np.concatenate([np.arange(-5,0),np.arange(1,6)]): #?-----
        #print(j)
        j=j*10
        pred=PC[(yr+j-2)*12:(yr+j)*12]
        prede.append(abs(pred-obs))        #todo-----

# 2.包络线-----
prede=np.array(prede).reshape(nmax*10,24)
prede_max=np.max(prede,axis=0)
prede_min=np.min(prede,axis=0)
prede_mean=np.mean(prede, axis=0)
print('prede_mean.shape',prede_mean.shape)

lprede=prede.reshape(nmax,10,24)
#lpb = np.argpartition([-lprede[i,:,11] for i in np.arange(0,15)], 5)[:5]   #?----- larger pb 前5个

lpb=[]
for i in np.arange(0,nmax):
    lpbi=np.argpartition(-lprede[i,:,23], 5)[:5]
    lpbi=lpbi+i*15
    lpb.append(lpbi)
lpb=np.array(lpb).reshape((nmax*5))   # nmax*5
np.savetxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_lpb_10.txt", lpb, fmt='%1.8f') #todo-----
print(lpb)

'''#3. growth rates of prediction error (nmax*40,11)
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
season_gr[:,0] = (prede[:, 2]-prede[:,0])/2
season_gr[:,1] = (prede[:, 5]-prede[:,2])/3 #春3、4、5的增长率
season_gr[:,2]= (prede[:, 8]-prede[:,5])/3
season_gr[:,3] = (prede[:, 11]-prede[:,8])/3 #秋9、10、11

#? 秋季-----
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

#? 春季-----
spr_column =season_gr[:, 1]  #?-----
row_max = np.max(season_gr, axis=1)
mask = (spr_column == row_max)
spb = np.where(mask)[0]   #Spring predictability barrier
#np.savetxt("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_spb.txt", spb, fmt='%1.8f') #todo-----

prede_spr=np.array([prede[i,:] for i in spb])
#print(prede_aut.shape)  (198,12)
prede_spr_max=np.max(prede_spr,axis=0)
prede_spr_min=np.min(prede_spr,axis=0)
prede_spr_mean=np.mean(prede_spr,axis=0)'''



'''#? plot: error growth--------------------------------
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
ax2.plot(x2,prede_spr_mean,c='g',lw=1.5,label='Spring', zorder=4)
ax2.plot(x2,prede_aut_mean,c='r',lw=1.5,label='Autumn', zorder=4)
ax2.fill_between(x2, prede_max, prede_min, color='DeepSkyBlue', alpha=1, zorder=1)
ax2.fill_between(x2, prede_spr_max, prede_spr_min, color='LimeGreen', alpha=0.6, zorder=2)
ax2.fill_between(x2, prede_aut_max, prede_aut_min, color='Salmon', alpha=0.8, zorder=3)
ax2.set_xlabel('month',size=16)
ax2.set_ylabel('prediction error',size=16)
plt.xticks(ticks=np.arange(0,12),labels=np.arange(1,13))
plt.yticks(ticks=np.arange(0,8))
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

plt.subplots_adjust(left=0.064,
                    bottom=0.098,
                    right=0.960,
                    top=0.915,
                    wspace=0.2,      #子图间水平距离
                    hspace=0.421     #子图间垂直距离
                   )
#CanESM2_index_prediction_error
plt.suptitle('NorCPM1', fontsize=25, x=0.5, y=0.983)   # x,y=0.5,0.98 (default)
plt.show()'''
#? ============================================================>





#todo (III) 前面是指数；后面开始变量
'''
# 有spb、apb的预测的初始误差结构（对其相应变量的初始误差进行EOF，并挑选高度相关的进行合成）；
# 15*40=600；divmod（spb[0],40(-20)）
# 哪个事件的哪次预报的，哪个变量
'''

#? 相关变量------------------------------------------------------------------->
#读取数据---------------------------
files = 'D:/decadal prediction/data/piControl/NorCPM1/tos*.nc'   
#files='F:/data/piControl/NorCPM1/tos*.nc'
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

# 筛选: spb、apb、lpb ----------
spb=np.loadtxt('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_spb.txt')
apb=np.loadtxt('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_apb.txt')
lpb=np.loadtxt('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_lpb_10.txt')
spby,spbx=divmod(spb,40)   #600=15*40-->spb.shape(year+x_pred)； float
spby = spby.astype(int)
spbx = spbx.astype(int)
#print(spbx)
#在事件前后筛选预测
spbx=spbx-5
spbx = [x + 1 if x >= 0 else x for x in spbx]   #?-----
#print(spbx)

apby,apbx=divmod(apb,40)   #600=15*40-->spb.shape(year+x_pred)； float
apby = apby.astype(int)
apbx = apbx.astype(int)
#在事件前后筛选预测
apbx=apbx-5
apbx = [x + 1 if x >= 0 else x for x in apbx]   #?-----

lpby,lpbx=divmod(lpb,10)   #600=15*40-->spb.shape(year+x_pred)； float
lpby = lpby.astype(int)
lpbx = lpbx.astype(int)
#在事件前后筛选预测
lpbx=lpbx-5
lpbx = [x + 1 if x >= 0 else x for x in lpbx]   #?-----


# 初始误差、变量区域(南纬20度以北的太平洋)   
#? 季节预报障碍的基础上，进一步分析到高度相关的初始误差----------------------------
#spby\spbx长度是一样的；y=spby[i];x=spbx[i];y是某个事件，x是波动年份
variable_p=ds.tos[:,70:160,120:280]   # sst pacific_related: 20S-70N, 120E-80W; PDO:20N-70N, 120E-100W
lon=variable_p.lon     #todo-----
lat=variable_p.lat
#print(lon,lat)

lenspb=len(spb); lenapb=len(apb); lenlpb=len(lpb)
#print(lenspb)
errors=np.empty((lenlpb,90,160))
#ij=np.empty((lenspb,2))
all_times = []
for i in np.arange(0,lenlpb):
    events = lpby[i]   #todo-----5
    yrs = lpbx[i]
    t1 = int((tquotient[events]-2) * 12)   #?   -7、-4、-1、0、2、5、8、11、14、17-----
    t2 = int((tquotient[events] + yrs*10-2) * 12)
    all_times.extend([t1, t2])  # 保留重复的时间点
preloaded_data = variable_p.isel(time=all_times).load()

for i in np.arange(0,lenlpb):      #todo -----
    idx1 = 2 * i      
    idx2 = 2 * i + 1  
    
    observation = preloaded_data.data[idx1]
    prediction = preloaded_data.data[idx2]
    
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

pc1=PC[:,0]
pc2=PC[:,1]
pc3=PC[:,2]
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
'''with open('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_spb_tos_errors12.txt', 'w') as f:
    # 遍历数组的每个切片
    for two_d_slice in errors:      #?---
        # 将二维切片转换为字符串，元素之间用空格分隔
        for row in two_d_slice:
            f.write(' '.join(map(str, row)) + '\n')
        # 每个切片之间添加一个空行
        f.write('\n')

with open('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_spb_tos_composition12.txt', 'w') as f:
    # 遍历数组的每个切片
    for two_d_slice in composition:
        # 将二维切片转换为字符串，元素之间用空格分隔
        for row in two_d_slice:
            f.write(' '.join(map(str, row)) + '\n')
        # 每个切片之间添加一个空行
        f.write('\n')

with open('D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_spb_tos_ppn12.txt', 'w') as f:
    # 遍历数组的每个切片
    for two_d_slice in ppn:
        # 将二维切片转换为字符串，元素之间用空格分隔
        for row in two_d_slice:
            f.write(' '.join(map(str, row)) + '\n')
        # 每个切片之间添加一个空行
        f.write('\n')
'''
print(np.nanmax(composition))

#? plot: initial error structure------------------
fig=plt.figure(figsize=(8,10),dpi=100) #

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

cmap1=cmaps.BlueDarkRed18                          #BlueDarkRed18 temp_diff_18lev 
list_cmap1=cmap1(np.linspace(0,1,10))  
cp=ListedColormap(list_cmap1,name='cp')

zt=[231,234,232,235,233,236]
an=['1+','1-','2+','2-','3+','3-']
for i in np.arange(0,6):
    ztn=zt[i]

    significant_mask=ppn[i,:,:]<0.05
    masked_data = np.where(significant_mask, composition[i,:,:], np.nan) #?-----

    proj=ccrs.PlateCarree(central_longitude=180)                                            
    ax=fig.add_subplot(ztn, projection=proj)                                                    # 之前一直在尝试subplots,但有说是subplots对ccrs.PlateCarree不起作用
    ax.contourf(lon, lat, masked_data , cmap=cp,transform=ccrs.PlateCarree(),levels=np.linspace(-3, 3, 11), extend="both")#, levels=np.linspace(-1, 1, 11)  )   #, levels=np.linspace(-4, 32, 10)
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.set_extent([120, 280, -20, 70], crs=ccrs.PlateCarree())                                 
    ax.coastlines(resolution="50m", linewidth=1)                                             # resolution 分辨率（valid scales）--110m、50m、10m。
    ax.tick_params(size=6, labelsize=16)
    #ax.set_xlabel('longitude',size=18)
    #ax.set_ylabel('latitude',size=18)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks([180,150,120,-90,-120,-150,-180],crs=ccrs.PlateCarree())                                    # [ ] 有特定经度就行，不在乎顺序
    ax.set_yticks( np.arange(-20,71,20), crs=ccrs.PlateCarree())
    ax.annotate(f'EOF{an[i]}',xy=(0.45,1.1),xycoords='axes fraction',c='k',fontsize=16,ha='left',va='top')

#绘制colorbar----------------------------------------
ax0=fig.add_axes([0.2,0.06,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-3, vmax=3)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cp),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('',  rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax                                         #调用colorbar的ax属性
#ax1.set_title('Temperature',pad=8, fontsize=16)
fc1.set_ticks(np.linspace(-3, 3, 11))
fc1.ax.tick_params(labelsize=15)      #调用colorbar的ax属性

#子图位置和标题-------------------------------------
plt.subplots_adjust(left=0.060,
                    bottom=0.150,
                    right=0.970,
                    top=0.92,
                    wspace=0.3,      #子图间水平距离
                    hspace=0.15     #子图间垂直距离
                   )
plt.suptitle('NorCPM1_lpb_tos_errors1_10', fontsize=20, x=0.5, y=0.98)# x,y=0.5,0.98(默认)
#plt.savefig('D:/decadal prediction/results/piControl/NorCPM1_tos_errors1.png')
plt.show()
#?==========================================>
    








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

