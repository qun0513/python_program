'''xarray总体数据结构：dataarray,dataset;
dadtset是多个dataarray的集合
dataararay可被认为是包含元数据的numpy数组'''
import xarray as xr
a=xr.open_dataset('')
a.to_netcdf()
'''data_vars:存储对应于数据变量的dataarray对象的有序字典
dims：保存多维数据每个维度的名称及长度的字典
coords：保存数据点在某个维度上的坐标的字典
attars：保存dataset的其他元数据的有序字典'''
#数据索引和选择
#基于位置.isel  基于标签.sel()
#where根据条件选取满足条件的格点值

