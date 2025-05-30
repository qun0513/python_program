import xarray as xr
#import matplotlib             #解决Linux无法可视化的问题
#matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray import DataArray

#数据读取————————————————————————————————————————————————————————————————————————————————————————
#U=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/U_1979-201908.nc')
U=xr.open_dataset('D:/GC/U_1979-201908.nc')
u=U.u.loc[:,500,90:40,0:359]
#print(u)

#SIC=xr.open_dataset('/home/dell/ZQ19/HadISST_ice_187001-202102.nc')
SIC=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
sic=SIC.sic.loc[:,89.5:39.5,-179.5:179.5]
sic=sic.isel(time=slice(1308,1796))          #1979-2020 (1979.1-2019.08)
#print(sic[0,15,135])
#Z=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/Z_1979-201908.nc')
Z=xr.open_dataset('D:/GC/Z_1979-201908.nc')
z=Z.z.loc[:,500,90:40,0:359]
z=z/10
#print(z)
lon=z.longitude
lat=z.latitude
lon=lon[::3]

#数据处理——————————————————————————————————————————————————————————————————————————————————————————
# U   纬向速度
#秋季
year=np.arange(1979,2019)
t1=u.sel(time=u.time.dt.month.isin([9]))
t2=u.sel(time=u.time.dt.month.isin([10]))
t3=u.sel(time=u.time.dt.month.isin([11]))
T1=t1.sel(time=t1.time.dt.year.isin([year]))
T2=t2.sel(time=t2.time.dt.year.isin([year]))
T3=t3.sel(time=t3.time.dt.year.isin([year]))
UA=(T1.data+T2.data+T3.data)/3
#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t4=u.sel(time=u.time.dt.month.isin([12]))
t5=u.sel(time=u.time.dt.month.isin([1]))
t6=u.sel(time=u.time.dt.month.isin([2]))
T4=t4.sel(time=t4.time.dt.year.isin([year1]))
T5=t5.sel(time=t5.time.dt.year.isin([year2]))
T6=t6.sel(time=t6.time.dt.year.isin([year2]))
UW=(T4.data+T5.data+T6.data)/3

# Z  位势高度场
#秋季
year=np.arange(1979,2019)
t7=z.sel(time=z.time.dt.month.isin([9]))
t8=z.sel(time=z.time.dt.month.isin([10]))
t9=z.sel(time=z.time.dt.month.isin([11]))
T7=t7.sel(time=t7.time.dt.year.isin([year]))
T8=t8.sel(time=t8.time.dt.year.isin([year]))
T9=t9.sel(time=t9.time.dt.year.isin([year]))
ZA=(T7.data+T8.data+T9.data)/3
#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t10=z.sel(time=z.time.dt.month.isin([12]))
t11=z.sel(time=z.time.dt.month.isin([1]))
t12=z.sel(time=z.time.dt.month.isin([2]))
T10=t10.sel(time=t10.time.dt.year.isin([year1]))
T11=t11.sel(time=t11.time.dt.year.isin([year2]))
T12=t12.sel(time=t12.time.dt.year.isin([year2]))
ZW=(T10.data+T11.data+T12.data)/3

# SIC  海冰密集度
#秋季
year=np.arange(1979,2019)
t13=sic.sel(time=sic.time.dt.month.isin([9]))
t14=sic.sel(time=sic.time.dt.month.isin([10]))
t15=sic.sel(time=sic.time.dt.month.isin([11]))
T13=t13.sel(time=t13.time.dt.year.isin([year]))
T14=t14.sel(time=t14.time.dt.year.isin([year]))
T15=t15.sel(time=t15.time.dt.year.isin([year]))
SA=(T13.data+T14.data+T15.data)/3

sa=np.empty((40,51,360))
for i in np.arange(0,360):
    if i<180:
        sa[:,:,i]=SA[:,:,180+i]
    if i>180:
        sa[:,:,i]=SA[:,:,i-180]

#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t16=sic.sel(time=sic.time.dt.month.isin([12]))
t17=sic.sel(time=sic.time.dt.month.isin([1]))
t18=sic.sel(time=sic.time.dt.month.isin([2]))
T16=t16.sel(time=t16.time.dt.year.isin([year1]))
T17=t17.sel(time=t17.time.dt.year.isin([year2]))
T18=t18.sel(time=t18.time.dt.year.isin([year2]))
SW=(T16.data+T17.data+T18.data)/3

sw=np.empty((40,51,360))
for i in np.arange(0,360):
    if i<180:
        sw[:,:,i]=SW[:,:,180+i]
    if i>180:
        sw[:,:,i]=SW[:,:,i-180]

#SVD分解——————————————————————————————————————————————————————————————————————————————————————--

time=np.arange(0,40)
#UA=UA[:,:,::5]
sa=sa[:,:,::3]
ZA=ZA[:,:,::3]
#UW=UW[:,:,::5]
sw=sw[:,:,::3]
ZW=ZW[:,:,::3]
#UA=DataArray(UA,coords=[time,lat,lon],dims=['time','lat','lon'])
sa=DataArray(sa,coords=[time,lat,lon],dims=['time','lat','lon'])
za=DataArray(ZA,coords=[time,lat,lon],dims=['time','lat','lon'])
#UW=DataArray(UW,coords=[time,lat,lon],dims=['time','lat','lon'])
sw=DataArray(sw,coords=[time,lat,lon],dims=['time','lat','lon'])
zw=DataArray(ZW,coords=[time,lat,lon],dims=['time','lat','lon'])

from xMCA import xMCA
svd=xMCA(sw,zw)
svd.solver()
lp, rp = svd.patterns(n=4)
le, re = svd.expansionCoefs(n=4)
frac = svd.covFracs(n=4)

#homogeneous 同性
#lho, rho, lphot, rphot = svd.homogeneousPatterns(n=3, statistical_test=True)
#heterogeneous 异性
#lhe, rhe, lphet, rphet = svd.heterogeneousPatterns(n=3, statistical_test=True)

print(frac.data)


#地图投影------------------------------------------------------------------
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import proplot as plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math

proj = plot.Proj('npstere', central_longitude=90)
fig,ax= plot.subplots(proj=proj)


ax.format(labels=False, grid=False, coast=True, metalinewidth=1.0, boundinglat=41)
#metalinewidth 外边框、 boundlinglat 最外围纬圈

#gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1, color='lightgrey', x_inline=True, y_inline=True,
#                  xlocs=np.arange(-180, 180,45)) #(-180, 180,45) 0,360, 45

ins = ax.inset_axes([0., 0., 1, 1], proj='polar', zorder=2)
ins.set_theta_zero_location("S", -90)

# 画经纬度label,很难自动生成：
def format_fn(tick_val, tick_pos):
    t = tick_val / math.pi / 2 * 360
    lons = LongitudeFormatter(zero_direction_label=False).__call__(t, tick_pos)
    return lons

formatter = mticker.FuncFormatter(format_fn)
ins.format( rlocator=[], facecolor='white', alpha=0, thetadir=1,
           thetalim=(0, 360), gridlabelpad=10
           # ,thetalocator=np.arange(0,360*math.pi/2,45*math.pi/2)#mticker.FixedLocator(np.arange(0,360,45))
           , thetaformatter=formatter  #LongitudeFormatter(zero_direction_label=False)
           , labelsize=30  # ,thetalabels=np.arange(0,360,45))
           )
 #rlocator 横向刻度

def mtick_gen(polar_axes, tick_depth, tick_degree_interval=45, color='r', direction='out', **kwargs):
    lim = polar_axes.get_ylim()
    radius = polar_axes.get_rmax()
    if direction == 'out':
        length = tick_depth / 100 * radius
    elif direction == 'in':
        length = -tick_depth / 100 * radius
    for angle in np.deg2rad(np.arange(0, 360, tick_degree_interval)):
        polar_axes.plot(
            [angle, angle],
            [radius + length, radius],
            zorder=500,
            color=color,
            clip_on=False, **kwargs,
        )
    polar_axes.set_ylim(lim)

mtick_gen(ins, 6, 45, color="k", lw=1, direction='out')
mtick_gen(ins, 4, 15, color="k", lw=1, direction='out')

ins.tick_params(labelsize=12)  #刻度值大小-----------------------------

#解决0度经线出现白条
from cartopy.util import add_cyclic_point
#左场（sic）
print(lp[0].shape)
print(lon.shape)
#sic1,lon=add_cyclic_point(lp[0],coord=lon)  #1
#sic2,lon=add_cyclic_point(lp[1],coord=lon)  #2
#sic3,lon=add_cyclic_point(lp[2],coord=lon)  #3
#右场（z）
#z1,lon=add_cyclic_point(rp[0],coord=lon)
#z2,lon=add_cyclic_point(rp[1],coord=lon)
#z3,lon=add_cyclic_point(rp[2],coord=lon)
#rho1,lon=add_cyclic_point(rho[0],coord=lon)

#sic4,lon=add_cyclic_point(lp[3],coord=lon)  #3
z4,lon=add_cyclic_point(rp[3],coord=lon)

#绘图-------------------------------------------------------------------

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueDarkRed18
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

plt.contourf(lon, lat, z4,cmap=cp)#,levels=np.linspace(-0.042,0.042,13))
#levels=np.linspace(-0.05,0.05,11)     levels=np.linspace(-0.02,0.02,9)
#MPL_PiYG_r\GMT_polar\BlueDarkRed18\GMT_panoply
plt.title('SVD_Winter_z4_Z500',loc='center',fontsize=12,pad=15)
cb = plt.colorbar(ax=ax, orientation="vertical", norm='normlize',pad=0.1, aspect=40, shrink=1)
cb.set_label(' ',size=3,rotation=0,labelpad=5,fontsize=12)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=10)
print("hello,world!")
plt.show()
#plt.savefig('D:/GC/SVD/SVD_Winter_z4_Z500.jpg')

#时间系数
fig,ax1=plt.subplots()
x=np.arange(1979,2019)
ax1.plot(x,le[3],c='r',label='sic')
ax1.plot(x,re[3],c='b',label='z')
ax1.tick_params(labelsize=10)
#ax1.set_xticks(np.arange(1979,2019,5))
#ax1.set_xticklabels(fontsize=15)
ax1.set_title('时间系数',fontsize=12)
ax1.tick_params(labelsize=12,size=6)
print("hello,world!")
plt.legend()
plt.show()


'''
rho1=np.array(rho1).reshape(51,121)
plt.contourf(lon, lat, rho1,cmap=cp,levels=np.linspace(-1,1,11))
#levels=np.linspace(-0.05,0.05,11)     levels=np.linspace(-0.02,0.02,9)
#MPL_PiYG_r\GMT_polar\BlueDarkRed18\GMT_panoply
plt.title('SVD_r1_homogeneous',loc='center',fontsize=12,pad=15)
cb = plt.colorbar(ax=ax, orientation="vertical", norm='normlize',pad=0.1, aspect=40, shrink=1)
cb.set_label(' ',size=3,rotation=0,labelpad=5,fontsize=12)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=10)
print("hello,world!")
plt.savefig('D:/GC/SVD/SVD_r1_homogeneous.jpg')
'''