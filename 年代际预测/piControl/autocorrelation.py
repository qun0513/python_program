 #? lag autocorrelation
 #? 批量读取数据、调整经度（0-360）、面积加权、向量化操作

import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import cmaps
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import math
import glob
import os
import dask
import cftime

'''
#读取数据----------------------------------------------------------------------------------------
files = 'F:/data/piControl/BCC-CSM2-MR/tos*.nc'  # 替换为你的文件夹路径
#files='D:/decadal prediction/data/piControl/NorCPM1/tos*.nc'
#file_list = sorted(glob.glob(files))  # 排序确保文件顺序正确

#xr.set_options(enable_cftimeindex=True)   # 全局设置使用 cftime 解析时间
combined_ds = xr.open_mfdataset(
    paths=files, 
    use_cftime=True,                  #时间解码
    #combine="by_coords",    # 按坐标自动对齐
    #parallel=True,                 # 启用并行读取
    #chunks={"time": 100},   # 分块处理大文件
    #engine="netcdf4"        # 指定引擎
)

#插值------------------
target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
regridder = xe.Regridder(combined_ds, target_grid, 'bilinear',
                         #, filename=None,
                         #periodic=True,                 # 处理经度周期性（如全球网格）
                         ignore_degenerate=True,  # 忽略退化的网格点#???某些网格点可能因为几何形状或数值问题导致插值算法无法正常处理
                         #extrap_method='nearest_s2d'  # 外推方法
                         )
regridderdata1= regridder(combined_ds)
print(combined_ds)
# (-180,180) 转为(0,360)------------------
regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
lat_1d = np.array(regridderdata1['lat'].isel(x=0).squeeze().data)    # 提取 NumPy 数组
lon_1d = np.array(regridderdata1['lon'].isel(y=0).squeeze().data)  # 提取 NumPy 数组
regridderdata1 = regridderdata1.assign_coords(lat=('y', lat_1d),lon=('x', lon_1d))
ds = regridderdata1.sortby('lon')
print(ds)
# remove the climatological annual cycle------------------
# 去除年循环（向量化操作）;列表不支持向量化操作，numpy可以
start=1440   #12*10
sst_nps = np.array(ds.tos[start:start+480,120:140,160:200].data).reshape((40, 12, 20,40))  #?-----  110:160,120:260
sst_ac = sst_nps.mean(axis=0)  # 直接计算年循环
sst3 = sst_nps - sst_ac              # 广播减法
sst4 = sst3.reshape((480, 20, 40)) # koe(20,65); np(50,150)
'''


#? HadISST、ERSST
#ds=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
ds=xr.open_dataset("D:/pycharm/sstv3b.mnmean_18540101-20200201.nc")

ds['lon'] = xr.where(ds['lon'] < 0, ds['lon'] + 360, ds['lon'])
ds1 = ds.sortby('lon')   # 确保经度是递增的
lon1=ds1.lon 
lat1=ds1.lat  
print(ds1)

glb=ds1.sst.sel(time=slice("1950-01-01", "2004-12-01"))
lon0=glb.lon  #.shape[1]
lat0=glb.lat
sst0=glb.data
length=len((glb.time))
yr=length//12
sst0[sst0==-1000]=np.nan   #HadISST -1000 and -1.8 需注意
sst0[sst0==-1.8]=np.nan

sst_g=sst0.reshape((yr,12,len(lat0),len(lon0)))  #1870-2018
sst_gm=np.nanmean(sst_g,axis=0)
sst_gma=sst_g-sst_gm

tos_np=ds1.sst.sel(time=slice("1950-01-01", "2004-12-01"),
                   lat=slice(46,38),
                   lon=slice(155,190))   #?---
print(tos_np)
sst1=tos_np.data
lat1=tos_np.lat
lon1=tos_np.lon

# remove the climatological annual cycle
sst_nps = sst1.reshape((yr, 12, len(lat1), len(lon1)))
sst_ac = sst_nps.mean(axis=0)  # 直接计算年循环
sst3 = sst_nps - sst_ac               # 广播减法

'''
global_means = np.nanmean(sst_gma.reshape(length, len(lat0),len(lon0)), axis=(1,2))
print(global_means.shape)
sst4 = sst3.reshape(length, len(lat1), len(lon1)) - global_means[:, None, None]
'''


#? area weight ---------------------------------------
'''
a = 6357000  
pi = np.pi
lat = np.array(ds.tos[0,110:160,0].lat.data)     #?-----

dx = 2* pi* a* np.cos(np.deg2rad(lat))/ 360  
dy = 2* pi* a/ 360  
s_grid = (dx[:, np.newaxis]* dy)* np.ones((140,))    #?------
valid_mask = ~np.isnan(sst4[0])  
total_area = np.sum(s_grid * valid_mask) 
weight = s_grid / total_area 

#sst4=abs(sst4)
#加权的海温距平
sst = np.nansum(sst4* weight[np.newaxis, :, :], axis=(1, 2))  # 结果形状: (6000,)
sst = sst.compute() if hasattr(sst, 'compute') else sst           # 处理Dask数组
'''
sst=np.nanmean(sst3.reshape(yr*12,len(lat1), len(lon1)),axis=(1,2))#.compute()

print('--------3')

#sst=np.nanmean(np.nanmean((sst10),axis=1),axis=1) #sst4.mean() or sst(weight area); sst10(8,9)


# lag autocoorelation---------------------------------------------------------
lag_ar=np.empty((12,12))
p=np.empty((12,12))
for i in np.arange(0,12):
    #print(i)
    for j in np.arange(0,12):
        if i+j+1>11:   #11*n
            lag_ar[i,j], p[i,j]=pearsonr(sst[i:39*12:12], sst[i+j+1:40*12:12])  #499\500, 68\69
        else:
            lag_ar[i,j], p[i,j]=pearsonr(sst[i:39*12:12], sst[i+j+1:39*12:12])  #?-----
    #print(i)


#print('lag_ar:',lag_ar)
#print('p:', p)

# plot ---------------------------------------------------------------------------------
x=np.arange(0,12)
y=np.arange(0,12)
X,Y=np.meshgrid(x,y)  #左下角
cmap1=cmaps.BlueDarkRed18                                        # BlueDarkRed18 temp_diff_18lev MPL_YlOrBr
list_cmap1=cmap1(np.linspace(0,1,20))
cp=ListedColormap(list_cmap1,name='cp')

fig=plt.figure(figsize=(10,8), dpi=100)
ax=fig.add_subplot()
contour_plot=ax.contour(X, Y, lag_ar, colors='k', levels=np.linspace(-1, 1, 21), linestyles='solid')
plt.contourf(X, Y, p, levels=[0, 0.05, 1], colors=['yellow', 'none']) 
plt.tick_params(labelsize=12, size=3)
plt.clabel(contour_plot, contour_plot.levels, inline=True,  fmt='%0.1f', fontsize=20)#,manual=[(i,4) for i in np.arange(0,12)])
plt.xticks(ticks=np.arange(0,12,1), labels=np.arange(1,13), size=25)
plt.yticks(ticks=np.arange(0,12,1), labels=np.arange(1,13), size=25)
ax.set_xlabel('lag time',size=26)
ax.set_ylabel('month',size=26)

plt.subplots_adjust(left=0.10,
                    bottom=0.100,
                    right=0.900,
                    top=0.900,
                    wspace=0.18,                 # horizontal distance between subgraphs
                    hspace=0.300                # vertical distance between subgraphs
                )

plt.suptitle('CWNP', fontsize=30, x=0.5, y=0.973)   # x,y=0.5,0.98 (default)
#plt.savefig("D:/decadal prediction/results/piControl/NorCPM1/NorCPM1_np_persistence.png")

plt.show()




#todo --------------------------------------------<
#我的面积加权的平均海温，改进版在上面
'''
# area weight ------------------------------------------------------------------------
lat=np.arange(30,50)
pi=3.14;a=6357000
dx=np.empty(65)     #x方向网格距
dy=np.empty(20)     #y方向网格距
s=np.empty(20)       #不同纬圈单个格点面积

for j in np.arange(0,19):
    dx[j]=2*pi*(a*math.cos((lat[j])*pi/180))*(1/360)   #不同纬圈周长的一份
    dy[j]=2*pi*a*(abs(lat[j]-lat[j+1])/360)                      #        经圈周长的一份
    s[j]=dx[j]*dy[j]
s[19]=2*pi*(a*math.cos((50)*pi/180))*(1/360)*2*pi*a*(1/360)  #第20个纬圈上，单个格点的面积

#计算海表温度
ss=0
for j in np.arange(0,20):
    x=0
    for k in np.arange(0,65):
        if not np.isnan(sst4[0, j, k]):
            x = x + 1    #各纬圈海洋格点个数
    ss=ss+s[j]*x         #海洋总表面积
print('---------3')
sst=np.empty(6000)
for i in np.arange(0,6000):
    sst0=0
    for j in np.arange(0,20):
        for k in np.arange(0,65):
            if not np.isnan(sst4[i, j, k]):
                sst41=sst4[i,j,k]           #有海表温度的格点
                sst0=sst0+sst41*(s[j]/ss)  #面积加权
    sst[i]=sst0       #面积加权后的海表温度                              -----
'''
#todo ------------------------------------>


#? hadley验证nino3、4区的持续性    
'''
ds=xr.open_dataset("D:/pycharm/HadISST_sst_187001-201903.nc")
#print(ds)
#print(data.sst[:,40:60,145:210])

# improved ---------
ds['longitude'] = xr.where(ds['longitude'] < 0, ds['longitude'] + 360, ds['longitude'])
sst7 = ds.sortby('longitude')   # 确保经度是递增的
'''

'''  ever
sst7=np.empty((432,180,360))      #（-180,180） 转为（0，360）------------------
for i in np.arange(0,360):
    if i<180:
        sst7[:,:,i]=ds.sst[1320:1752,:,180+i]   #1950-2018\1980-2018 ~ 828\468~960\1320
    if i>=180:
        sst7[:,:,i]=ds.sst[1320:1752,:,i-180]   #1980-2015(05); 960:1752(1632)
'''

'''
# improved
# remove the global-mean SST anomaly------------------
sst_g=sst7.reshape((36,12,180,360))        # global
sst_gac=np.nanmean(sst_g, axis=0)         # global annual cycle
sst_gaca=np.empty((36,12,180,360))      # global annual cycle anomaly
#remove the climatological annual cycle------------------------
sst_np1=sst7[:,85:95,160:210]                          # KOE: 30°–50°N,145°E–150°W; nino3：85:95(-5~5),210:270(150-90W); nino4: 85:95(-5~5),160:210(160E-150W)
sst_nps1=sst_np1.reshape((36,12,10,50))          #年循环
sst_ac1=np.nanmean(sst_nps1,axis=0)             #年循环气候态

sst8=np.empty((36,12,10,50))
sst9=np.empty((36,12,10,50))
for i in np.arange(0,36):
    for j in np.arange(0,12):
        
        #piControl 不需要去除全球变暖趋势
        #sst_gaca[i,j,:,:]=sst_g[i,j,:,:]-sst_gac[j,:,:]
        #sst_gma=np.nanmean(sst_gaca[i,j,:,:])        # global-mean anomaly
        #sst8[i,j,:,:]=sst_nps1[i,j,:,:] - sst_gma            # subtract a sequence
        #sst9[i,j,:,:]=sst8[i,j,:,:]-sst_ac1[j,:,:]                # subtract a matrix

        sst8[i,j,:,:]=sst_nps1[i,j,:,:]-sst_ac1[j,:,:]
sst10=sst8.reshape((432,10,50))   #去趋势与否，年限。
'''

'''
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

lag_ar=np.empty((499,12,12))
for i in np.arange(0,499):
    for j in np.arange(0,12):
        for k in np.arange(0,12):
            if k==11:
                lag_ar[i,j,k]=calculate_correlation(tos_np[i,j,:,:],tos_np[i+1,j,:,:])
            else:
                lag_ar[i,j,k]=calculate_correlation(tos_np[i,j,:,:],tos_np[i,k+1,:,:])
lagar=np.mean(lag_ar, axis=0)
#print(lagar)
'''

#todo  1*1度  -------------------------------------------<
''' 
# remove the climatological annual cycle------------------
sst_nps=tos_koe.reshape((500,12,20,65))     # np sst 
sst_ac=np.nanmean(sst_nps, axis=0)             # annual cycle

sst3=np.empty((500,12,20,65))
for i in np.arange(0,500):
    for j in np.arange(0,12):
        sst3[i,j,:,:]=tos_koe[i,j,:,:]-sst_ac[j,:,:]
sst4=sst3.reshape((6000,20,65))
print('-----------3')

# area weight ------------------------------------------------------------------------
lat=np.arange(30,50)
pi=3.14;a=6357000
dx=np.empty(65)     #x方向网格距
dy=np.empty(20)     #y方向网格距
s=np.empty(20)       #不同纬圈单个格点面积

for j in np.arange(0,19):
    dx[j]=2*pi*(a*math.cos((lat[j])*pi/180))*(1/360)   #不同纬圈周长的一份
    dy[j]=2*pi*a*(abs(lat[j]-lat[j+1])/360)                      #        经圈周长的一份
    s[j]=dx[j]*dy[j]
s[19]=2*pi*(a*math.cos((50)*pi/180))*(1/360)*2*pi*a*(1/360)  #第20个纬圈上，单个格点的面积

#计算海表温度
ss=0
for j in np.arange(0,20):
    x=0
    for k in np.arange(0,65):
        if not np.isnan(sst4[0, j, k]):
            x = x + 1    #各纬圈海洋格点个数
    ss=ss+s[j]*x         #海洋总表面积

sst=np.empty(6000)
for i in np.arange(0,6000):
    sst0=0
    for j in np.arange(0,20):
        for k in np.arange(0,65):
            if not np.isnan(sst4[i, j, k]):
                sst41=sst4[i,j,k]           #有海表温度的格点
                sst0=sst0+sst41*(s[j]/ss)  #面积加权
    sst[i]=sst0       #面积加权后的海表温度                              -----
print('sst:',sst,sst.shape)
'''
#todo --------------------------------------------------->


#? HadGEM3-GC31-MM   (0.25,0.25)---------------------------------------------------------<
"""
#插值------------------
target_grid = xe.util.grid_global(0.25, 0.25)                   #定义目标网格
#regridder = xe.Regridder(tos1, target_grid, 'bilinear', filename=None)
#regridderdata1= regridder(tos1['tos'])
#regridderdata2= regridder(tos2['tos'])
#regridderdata3= regridder(tos3['tos'])
#regridderdata4= regridder(tos4['tos'])
#regridderdata5= regridder(tos5['tos'])
regridder = xe.Regridder(tos1, target_grid, 'bilinear',
                         #, filename=None,
                         #periodic=True,                 # 处理经度周期性（如全球网格）
                         ignore_degenerate=True,  # 忽略退化的网格点#???某些网格点可能因为几何形状或数值问题导致插值算法无法正常处理
                         #extrap_method='nearest_s2d'  # 外推方法
                         )
regridderdata1= regridder(tos1)
#print('----------------1','\n',regridderdata1)
#print(regridderdata1.tos[:,500:501,700:701])

# (-180,180) 转为(0,360)------------------
regridderdata1['lon'] = xr.where(regridderdata1['lon'] < 0, regridderdata1['lon'] + 360, regridderdata1['lon'])
ds = regridderdata1.sortby('lon')  # 确保经度是递增的; if 经度和纬度是二维的 (y, x) 格式，
#print('regridderdata1------------------','\n',ds)
#print(ds[:,500:501,700:701])


'''#我的这个算法计算量太大了  -180-180-->0-360
xdata=[regridderdata1]#,regridderdata2,regridderdata3,regridderdata4,regridderdata5]
tos0=[]
#combined=np.ones((1,180*4,360*4))
for j in np.arange(0,1):
    regridderdata=xdata[j]
    rdata=np.empty(regridderdata.shape)
    print(j)
    for i in np.arange(0,360*4):
        if i<180*4:
            rdata[:,:,i]=regridderdata[:,:,180*4+i]
        if i>=180*4:
            rdata[:,:,i]=regridderdata[:,:,i-180*4]
        print(i)
    #combined = np.concatenate([combined,np.array(rdata)], axis=0)  #解决append时必须等长的问题，concatenate则不需要
    #tos0.append(rdata[:,:,:])
 '''
print('--------2')

# North Pacific------------------
#lon=regridderdata1.lon[110:160,110:260]
#lat=regridderdata1.lat[110:160,110:260]
#tos0=combined[1:6001,:,:]   #
tos=np.array(ds).reshape(500,12,720,1440)
#tos_np=tos[:,:,110:160,110:260]                       # 500 year, north pacific
tos_koe=tos[:,:,480:560,580:840]                     #30°–50°N,145°E–150°W, (20,65), KOE
#print("tos_koe:",tos_koe[0,0,:,:])
print('tos_koe:','\n',tos_koe)
# improved =====, 更多细节参照piControl(PDO+EEMD).py

# remove the climatological annual cycle------------------
sst_nps=tos_koe.reshape((500,12,80,260))     # np sst 
sst_ac=np.nanmean(sst_nps, axis=0)             # annual cycle

sst3=np.empty((500,12,80,260))
for i in np.arange(0,500):
    for j in np.arange(0,12):
        sst3[i,j,:,:]=tos_koe[i,j,:,:]-sst_ac[j,:,:]
sst4=sst3.reshape((6000,80,2604))
print('-----------3')

# area weight ------------------------------------------------------------------------
lat=np.arange(30,50,0.25)
pi=3.14;a=6357000
dx=np.empty(260)     #x方向网格距
dy=np.empty(80)     #y方向网格距
s=np.empty(80)       #不同纬圈单个格点面积

for j in np.arange(0,79):
    dx[j]=2*pi*(a*math.cos((lat[j])*pi/180))*(1/360/4)   #不同纬圈周长的一份
    dy[j]=2*pi*a*(abs(lat[j]-lat[j+1])/360)                      #        经圈周长的一份
    s[j]=dx[j]*dy[j]
s[79]=2*pi*(a*math.cos((50)*pi/180))*(1/360/4)*2*pi*a*(1/4/360)  #第20个纬圈上，单个格点的面积

#计算海表温度
ss=0
for j in np.arange(0,80):
    x=0
    for k in np.arange(0,260):
        if not np.isnan(sst4[0, j, k]):
            x = x + 1    #各纬圈海洋格点个数
    ss=ss+s[j]*x         #海洋总表面积

sst=np.empty(6000)
for i in np.arange(0,6000):
    sst0=0
    for j in np.arange(0,80):
        for k in np.arange(0,260):
            if not np.isnan(sst4[i, j, k]):
                sst41=sst4[i,j,k]           #有海表温度的格点
                sst0=sst0+sst41*(s[j]/ss)  #面积加权
    sst[i]=sst0       #面积加权后的海表温度                              -----
print('sst:',sst,sst.shape)
"""
#?---------------------------------------------------------------------------------------------->


#?---
'''# 预生成索引数组
n_years = 500
months = 12
total_samples = n_years * months

# 初始化结果数组
lag_ar = np.empty((12, 12))
p = np.empty((12, 12))
print(1)
for lead in range(12):
    for month in range(12):
        start1 = month
        end1 = total_samples - (lead + 1)
        start2 = month + lead + 1
        end2 = total_samples
        
        x = sst[start1:end1:12]
        y = sst[start2:end2:12]
        
        lag_ar[month, lead], p[month, lead] = pearsonr(x, y)
        print(month)
    print(lead)'''
