#数据读取---------------------------------------------------------------
import os
import xarray as xr
import numpy as np
import math
import  matplotlib.pyplot as plt
path = 'D:/dyhl/data_60_89_CTL'         #控制实验
#path = 'D:/dyhl/data_60_89_Wind'        #风应力实验
#path = 'D:/dyhl/data_60_89_Water'       #淡水通量实验

ff=[]
for filename in os.listdir(path):
    full_path = os.path.join(path,filename)
    f = xr.open_dataset(full_path,decode_times=False)
    ff.append(f.ts[0,0,:,:])     #变量
fff=np.array(ff).reshape(360,115,182)
#print(f.us.data)


#print(fff)
print(f)

#f = xr.open_dataset("D:/dyhl/data_60_89_CTL/MMEAN0060-01.nc",decode_times=False) #LICOM: LASG/IAP Climate system Ocean Model

# z0: sea surface height; hi: thickness of ice
# hd: thickness of ice in one grid;
# ic1: total of number of levels involved in convection per day;
# ic2: number of levels ventilated per day;
# net1: net surface heat  flux; net2: net surface salt flux;---------------------
# mld: mixed layer depth;------------
# akm: turbulent vertical viscosity
# akt: turbulent heat vertical viscosity
# aks: turbulent salt vertical viscosity (m^2/s)
# ts: temperature; ss: salinity (psu);--------------------
# us: zonal current; vs: meridional current; ws: vertical current (m/s)----------------------
# su: Uwindstress; sv: Vwindstress (Pa);---------------
# lthf: latent heat flux; sshf: sensible heat flux (w/m**2);------------------
# lwv: Longwave; swv: Shortwave (w/m**2);
# psi:Meridioanl Stream Function; bsf: Barotropic Stream Function (Sverdrup).

# 笔记 1 ----------------------------------------------------------------------------
'''
如果读取nc文件时程序报错“unable to decode time units..."，即无法解析时间单位时，
原因可能在于该nc文件是模式输出文件，时间通常是从0开始的，导致xarray模块无法解析，解决方案如下：
f=xr.open_dataset("xxx.nc",decode_times=False) #添加decode_times=False
'''
#-----------------------------------
'''
xr.contact: 将多个数组沿其维度串联成一个更大的数组。
xr.merge: 将多个不同的数组合并成一个数据集。
'''

lat=f.ts.lat.values     #values/ data
#print(lat)

