import xarray as xr


'''
a=xr.open_dataset("D:/decadal prediction/data/piControl/BCC-CSM2-MR/ua_Amon_BCC-CSM2-MR_piControl_r1i1p1f1_gn_185001-188912.nc")
b=xr.open_dataset("D:/decadal prediction/data/piControl/CanESM2/ua_Amon_CanESM2_piControl_r1i1p1_201501-231012.nc")
c=xr.open_dataset("D:/decadal prediction/data/piControl/HadGEM3-GC31-MM(LL)/ua_Amon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_195001-204912.nc")
c1=xr.open_dataset("D:/decadal prediction/data/piControl/HadGEM3-GC31-MM(LL)/ua_Amon_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn_185001-186912.nc")
d=xr.open_dataset("D:/decadal prediction/data/piControl/MIROC6/ua_Amon_MIROC6_piControl_r1i1p1f1_gn_320001-320912.nc")
e=xr.open_dataset("D:/decadal prediction/data/piControl/MRI-ESM2-0/ua_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-189912.nc")
f=xr.open_dataset("D:/decadal prediction/data/piControl/NorCPM1/ua_Amon_NorCPM1_piControl_r1i1p1f1_gn_000101-010012.nc")

print(1,a)
print(2,b)
print(3,c)
print(4,c1)
print(5,d)
print(6,e)
print(7,f)
'''

"""
from tqdm import tqdm
import numpy as np
for _ in tqdm(range(1000)):
    x=np.arange(2,10)
    surr = np.zeros_like(x)
print(surr)
"""

"""
import numpy as np
from PyEMD import EEMD

# 定义计算零点数的函数
def count_zero_crossings(signal):
    count = 0
    for i in range(1, len(signal)):
        if (signal[i] > 0 and signal[i-1] < 0) or (signal[i] < 0 and signal[i-1] > 0):
            count += 1
    return count

# 定义计算周期的函数
def calculate_period(zero_crossings, total_time):
    if zero_crossings == 0:
        return np.inf  # 避免除以零
    return (total_time * 2) / zero_crossings

# 生成示例信号
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

# 进行EEMD分解
eemd = EEMD()
imfs = eemd.eemd(signal)

# 计算每个IMF的零点数和周期
total_time = t[-1] - t[0]
zero_crossings = {}
periods = {}

for i, imf in enumerate(imfs):
    zero_crossings[f"IMF_{i+1}"] = count_zero_crossings(imf)
    periods[f"IMF_{i+1}"] = calculate_period(zero_crossings[f"IMF_{i+1}"], total_time)

# 输出结果
for imf, count in zero_crossings.items():
    print(f"{imf}: {count} zero crossings")

for imf, period in periods.items():
    print(f"{imf}: {period} seconds period")
"""

import numpy as np
from scipy.stats import pearsonr

a=np.arange(0,9).reshape((3,3))
b=np.arange(1,10).reshape((3,3))

def calculate_correlation(matrix1, matrix2):
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()

    # 找出非 NaN 的位置
    valid = ~np.isnan(flat1) & ~np.isnan(flat2)
    
    # 如果有效数据点足够多，则计算相关系数
    if np.sum(valid) >= 2:  # 至少需要两个点计算相关系数
        corr, _ = pearsonr(flat1[valid], flat2[valid])
    else:
        corr = np.nan  # 如果有效数据点不足，则返回 NaN
    
    return corr
print(calculate_correlation(a,b))
