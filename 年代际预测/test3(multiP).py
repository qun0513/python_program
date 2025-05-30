from multiprocessing import Pool
import numpy as np

def f(x):
    if x == 0:
        data = np.arange(0, 100).reshape(10, 10)
    elif x == 1:
        data = np.arange(0, 900).reshape(30, 30)
    elif x == 2:
        data = np.arange(0, 400).reshape(20, 20)
    y=x**2
    # 使用 f-string 格式化文件名
    #np.savetxt(f'multi{x}{y}.txt', data,delimiter='  ',fmt='%.2f')
    # 如果您需要返回文件名，可以取消注释下一行
    # return filename

if __name__ == '__main__':
    with Pool(3) as p:
        # 如果您需要收集返回值，可以使用 p.starmap 并提供一个包含参数和函数的元组列表
        # p.starmap(f, [(i,) for i in np.arange(0, 3)])
        p.map(f, np.arange(0, 3))


import xarray as xr
import xesmf as xe
data=xr.open_dataset("D:/decadal prediction/data/hindcast/MIROC6/tos_Omon_MIROC6_dcppA-hindcast_s1960-r1i1p1f1_gn_196011-197012.nc")
        
target_grid = xe.util.grid_global(1, 1)                   #定义目标网格
regridder = xe.Regridder(data, target_grid, 'bilinear', filename=None)
regridderdata= regridder(data['tos'])
print(regridderdata)
rdata=np.empty(regridderdata.shape)                                               #（-180,180） 转为（0，360）------------------
for i in np.arange(0,360):
    if i<180:
        rdata[:,:,i]=regridderdata[:,:,180+i]
    if i>180:
        rdata[:,:,i]=regridderdata[:,:,i-180]
lon=regridderdata.lon[110:160,110:260]                          #  也很关键，要理解数据定位的区域 和 画布定位的区域，如何让二者重合
lat=regridderdata.lat[110:160,110:260]                            #    经纬度都挑选那个区域
print(lat)
print(lat.data[:,0])


