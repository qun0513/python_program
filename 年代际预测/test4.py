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

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

'''
# 数据
values1=[4,8,5,3]
values2=[1,9,5,5]
values1=[4,7,6,3]
values2=[4,10,5,2]
values1=[3,6,7,7]
values2=[2,6,7,5]

# 设置柱状图的宽度和位置
bar_height = 0.2
index = np.arange(0,4)

# 创建图形和坐标轴
fig, ax = plt.subplots()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus'] = False              #用来正常显示负号

# 绘制柱状图，设置颜色和边框
bars1 = ax.bar(index-bar_height/2, values1, bar_height, label='暖事件', color='r', edgecolor='black')
bars2 = ax.bar(index + bar_height/2, values2, bar_height, label='冷事件', color='b', edgecolor='black')

# 添加标题、轴标签和图例
plt.xticks(ticks=[0,1,2,3],labels=['春季','夏季','秋季','冬季'],size=16)
#ax.set_xlabel('季节')
plt.yticks(ticks=np.arange(0,12,2),labels=np.arange(1,13,2),size=16)
ax.set_ylabel('次数',size=16)
ax.set_title('HadISST',fontsize=20)
ax.legend(fontsize=16)

# 显示图形
plt.tight_layout()
plt.show()'''



import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 6 * np.pi, 1000)
y = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='cos(x)')

first_peak_index = np.argmax(y[:len(y)//3])  # 假设第一个峰值在前1/3的位置
second_peak_index = np.argmax(y[len(y)//3:2*len(y)//3]) + len(y)//3  # 假设第二个峰值在中间1/3的位置

shift = 2 * np.pi
shifted_x = x[first_peak_index:second_peak_index] + shift
shifted_y = np.cos(shifted_x)

plt.plot(shifted_x+2*np.pi, shifted_y, color='red', linewidth=3)

shifted_start_x = shifted_x[0]
shifted_end_x = shifted_x[-1]
green_start_x = shifted_start_x - 0.25 * 2 * np.pi  # 提前红色线四分之一个周期
green_end_x = shifted_end_x - 0.25 * 2 * np.pi  # 提前红色线四分之一个周期

green_x = x[(x >= green_start_x) & (x <= green_end_x)]
green_y = np.cos(green_x)

plt.plot(green_x, green_y, color='green', linewidth=3)

plt.xlabel('x', size=16)
plt.ylabel('cos(x)', size=16)
plt.tick_params(axis='both', size=3, labelsize=16)
plt.grid(True)
plt.legend()
plt.show()


'''a=np.loadtxt("C:/Users/Zhao Qun/Downloads/CanESM2_tos_errors_3.txt")
x = a.size // (90 * 160)
a=a.reshape((x,90,160))

print(a.shape)
a=np.loadtxt("C:/Users/Zhao Qun/Downloads/MIROC6_tos_errors_1.txt")
x = a.size // (90 * 160)
a=a.reshape((x,90,160))

print(a.shape)'''

'''x=np.array([5,6,7])
y=x+1*12
print(y)


def find_numbers_with_multiple_of_10_difference(target_number):
    result = []
    for num in range(0, 490):
        if num != target_number and abs(num - target_number) % 10 == 0:
            result.append(num)
    return result

# 示例：找出与100相差10的倍数的所有数
for target in np.arange(90,91):
    result = find_numbers_with_multiple_of_10_difference(target)
    #print(f"与 {target} 相差10的倍数的所有数（0-500）：{result}")
    print(np.array(result))'''

'''def find_adjacent_periods(data, target_start_idx, target_end_idx, window_years=10):
    adjacent_periods = []
    steps_per_year = 12
    half_window = window_years // 2

    for i in range(len(data)):
        # 确定当前索引对应的时间范围
        current_start_year = i // steps_per_year
        current_start_month = i % steps_per_year + 1
        current_end_year = (i + steps_per_year * window_years - 1) // steps_per_year
        current_end_month = (i + steps_per_year * window_years - 1) % steps_per_year + 1

        # 检查当前窗口是否与目标窗口相邻
        # 这里的相邻可以定义为目标窗口的前后各一个窗口
        if ((current_start_year + current_start_month / steps_per_year >= target_start_idx / steps_per_year - half_window - 1) and
            (current_end_year + current_end_month / steps_per_year <= target_end_idx / steps_per_year + half_window + 1) and
            (abs(current_start_year - target_start_idx / steps_per_year) <= half_window + 1 or
             abs(current_end_year - target_end_idx / steps_per_year) <= half_window + 1)):
            adjacent_periods.append((current_start_year, current_end_year))

    return adjacent_periods

# 创建模拟的 500 年 * 12 月时间序列数据
np.random.seed(42)
data = np.random.rand(500 * 12)

# 目标时间段索引（从 13 年到 23 年）
target_start_idx = 13 * 12  # 第 13 年的第一个月的索引
target_end_idx = 23 * 12 - 1  # 第 23 年的最后一个的索引（假设每年有 12 个月）

# 寻找目标时间段前后的其他 10 年时间段
nearby_periods = find_adjacent_periods(data, target_start_idx, target_end_idx)

# 打印结果
print("目标时间段：13 - 23 年")
print("相邻的 10 年时间段：")
for period in nearby_periods:
    print(f"{period[0]} - {period[1]} 年")'''

'''nmax=15  #挑选15个事件
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
    ax.scatter(top_indices,top_values,c='r', marker='o',s=100,label='maximum value')  #240: 20year*12
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
'''

'''a=np.concatenate([np.arange(0,3),np.arange(0,2),np.arange(0,4)])
print(a)'''

'''
model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
for i in np.arange(0,6):   #  model*6
    mdl=model[i]
    files=f'F:/data/piControl/{mdl}/mlotst*.nc'
    combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
    #print(combined_ds)
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
    
'''




"""x='x'
print(x,'----------------')
month=[0,2,5,8,11]
for k in month:
    print(k)"""


'''
model=['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1']
for i in np.arange(0,1):
    #? 读取数据
    mdl=model[i]
    variables=['tos','zg','va','thetao']   #u[0]、v[0]、tos放一个循环里；thetao：5，9，13level
    for j in np.arange(2,3):
        variable=variables[j]
        files=f'F:/data/piControl/{mdl}/{variable}*.nc'
        combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
    print(combined_ds.isel(plev=0))
    # lev可能得手动筛选
'''

'''
variables=['tos','zg','va','thetao']   #u[0]、v[0]、tos放一个循环里；thetao：5，9，13level
for j in np.arange(0,4):
    variable=variables[j]
    files=f'F:/data/piControl/NorCPM1/{variable}*.nc'
    combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)
    if j==1 or j==2:
        combined_ds=combined_ds[variable][:,0,:,:]
    if j==3:
        for jj in [5]:
            combined_ds=combined_ds[variable][:,5,:,:]
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
'''

    #print(ds)

'''a=xr.open_dataset('F:/data/piControl/BCC-CSM2-MR/thetao_Omon_BCC-CSM2-MR_piControl_r1i1p1f1_gn_185001-185912.nc')
print(a.thetao[:,13,:,:]) #5,9,13
b=xr.open_dataset('F:/data/piControl/BCC-CSM2-MR/ua_Amon_BCC-CSM2-MR_piControl_r1i1p1f1_gn_185001-188912.nc')
print(b.ua[:,0,:,:])
c=xr.open_dataset('F:/data/piControl/BCC-CSM2-MR/va_Amon_BCC-CSM2-MR_piControl_r1i1p1f1_gn_185001-188912.nc')
print(c.va[:,0,:,:])'''

"""print(1%12,1/12,1//12)
x=np.ones((3,3,3))
y=np.arange(0,27).reshape((3,3,3))
print(y-x)"""

'''ds=xr.open_dataset("D:/pycharm/HadISST_sst.nc")
print(ds.time)'''

'''ds=xr.open_dataset("D:/pycharm/ersst.nc")
print(ds.attrs)
files='F:/data/piControl/NorCPM1/tos*.nc'
combined_ds = xr.open_mfdataset(paths=files, use_cftime=True)


target_grid = xe.util.grid_global(1, 1)                   
regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                         ignore_degenerate=True)
regridderdata1= regridder(combined_ds)


regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
lat_1d = np.array(regridderdata1['lat'].isel(x=0).squeeze().data)  
lon_1d = np.array(regridderdata1['lon'].isel(y=0).squeeze().data)  
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
ds = regridderdata1.sortby('lon')

fig=plt.figure(figsize=(10,8),dpi=100)
x=np.arange(12)
proj=ccrs.PlateCarree(central_longitude=180)                                            
ax=fig.add_subplot(211,projection=proj) 

ax.set_extent([120, 280, -20, 70], crs=ccrs.PlateCarree())
ax.coastlines(resolution="50m", linewidth=1)

ax0=fig.add_axes([0.2,0.06,0.6,0.03])
norm =mpl.colors.Normalize(vmin=-3, vmax=3)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap='rainbow'),cax=ax0,
                 orientation='horizontal',extend='both')
fc1.set_label('',  rotation=0, labelpad=5, fontsize=16,loc='center')
ax1=fc1.ax
plt.savefig('111.png')





print('Hello,world!')'''