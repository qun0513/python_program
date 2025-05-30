import xarray as xr
import matplotlib             #解决Linux无法可视化的问题
matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray import DataArray


U=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/U_1979-201908.nc')
#U=xr.open_dataset('D:/GC/U_1979-201908.nc')
u=U.u.loc[:,500,90:40,0:359]
#print(u)

SIC=xr.open_dataset('/home/dell/ZQ19/HadISST_ice_187001-202102.nc')
#SIC=xr.open_dataset('D:/GC/HadISST_ice_187001-202102.nc')
sic=SIC.sic.loc[:,89.5:39.5,-179.5:179.5]
sic=sic.isel(time=slice(1308,1796))          #1979-2020 (1979.1-2019.08)

Z=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/Z_1979-201908.nc')
#Z=xr.open_dataset('D:/GC/Z_1979-201908.nc')
z=Z.z.loc[:,500,90:40,0:359]
#print(z)
z=z/10

lon=z.longitude
lat=z.latitude
lon=lon[::10]



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

ua=np.array(UA).reshape(40,18360)
uw=np.array(UW).reshape(40,18360)
ua=np.transpose(ua)                #矩阵转置
uw=np.transpose(uw)
#print('UA','\n',UA)
#print('ua','\n',ua)
ua=ua[::10]                        #1836*40
uw=uw[::10]
print(ua.shape)


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

#print('T10','\n',T10)

za=np.array(ZA).reshape(40,18360)
zw=np.array(ZW).reshape(40,18360)
za=np.transpose(za)
zw=np.transpose(zw)
za=za[::10]                        #1836*40
zw=zw[::10]
print(za.shape)


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

#print('T6','\n',T6)
#print(UW.shape)

sa=np.array(SA).reshape(40,18360)
sw=np.array(SW).reshape(40,18360)
sa=np.transpose(sa)                #矩阵转置
sw=np.transpose(sw)
sa=sa[::10]                        #1836*40
sw=sw[::10]
print(sa.shape)
sa=np.nan_to_num(sa)               #处理nan、infs

#去趋势
#import scipy
#from scipy import signal
#emon12=scipy.signal.detrend(emon_12)    #去趋势
#print(emon12)


#标准化
# 1
'''
import torch
import torch.nn.functional as F
ua=torch.Tensor(ua)
za=torch.Tensor(za)
ua=F.softmax(ua)
za=F.softmax(za)
'''
# 2

#import sklearn
from sklearn.preprocessing import scale
za=scale(za)
ua=scale(ua)
sa=scale(sa)

# 3
'''
import math
def exponential(x):
    m_value=x.max()
    return math.log10(x)/math.log10(m_value)
ua=exponential(ua)
za=exponential(za)
'''
#print('za','\n',za)
#print(za.max(),za.min())
#print('ua','\n',ua)
#print(ua.max(),ua.min())

#协方差矩阵

ca=np.dot(ua,np.transpose(za))/40     #秋季  (1836,40)*(40,1836)
cw=np.dot(sw,np.transpose(zw))/40     #冬季
print('ca','\n',ca)
#print('cw','\n',cw)
#print('ca','\n',ca)                #1836*1836

#ca=np.nan_to_num(ca)               #处理nan、infs
#cw=np.nan_to_num(cw)


#奇异值分解
#3
'''
from sklearn.decomposition import TruncatedSVD
trun_svd=TruncatedSVD(n_components=2)
a=trun_svd.fit_transform(ca)
print('a','\n',a.size)
#print('s','\n',s)
#print('n','\n',n)
'''

# 2
'''
#SA=np.nan_to_num(SA)
UA=DataArray(UA,dims=['time','lat','lon'])
SA=DataArray(SA,dims=['time','lat','lon'])
ZA=DataArray(ZA,dims=['time','lat','lon'])
print(UA)
print(SA.shape)
print(ZA.shape)
from xMCA import xMCA
u=u[:,:,::10]
z=z[:,:,::10]
sic=sic[:,:,::10]
svd=xMCA(sic,z)
svd.solver()
lp, rp = svd.patterns(n=2)
le, re = svd.expansionCoefs(n=2)
frac = svd.covFracs(n=2)
print(frac.data)
'''

# 1

#from scipy import linalg
m,s,n=np.linalg.svd(ca)
n=np.transpose(n)
#print(m.shape)      #1836*1836
#print(s.shape)      #1836
#print(n.shape)      #1836*1836

#解释方差
ss=0
sss=np.empty((3,1))
for i in np.arange(1,1836):
    ss=ss+s[i]**2
for i in np.arange(1,4):
    sss[i-1,0]=s[i]**2/ss
print('sss','\n',sss)
print(s)
print(s[0])
print(s[1])
print(s[2])
print(s[3])
#第一模态--Autumn
#ca1=m[:,0]*s[0]*np.transpose(n[:,0])        #(1836,1)*(1,1)*(1,1836)
#ca1=np.dot(np.dot(m[:,0],s[0]),np.transpose(n[:,0]))
m1=np.array(m[:,0]).reshape(1836,1);m2=np.array(m[:,1]).reshape(1836,1);m3=np.array(m[:,2]).reshape(1836,1)
s1=np.array(s[0]).reshape(1,1);s2=np.array(s[1]).reshape(1,1);s3=np.array(s[2]).reshape(1,1)
n1=np.array(n[:,0]).reshape(1836,1);n2=np.array(n[:,1]).reshape(1836,1);n3=np.array(n[:,2]).reshape(1836,1)
m4=np.array(m[:,3]).reshape(1836,1)
s4=np.array(s[3]).reshape(1,1)
n4=np.array(n[:,3]).reshape(1836,1)
#print('m[:,0]',m[:,0].shape)
#print('s[0]',s[0].shape)
#print('np.transpose(n[:,0])',np.transpose(n[:,0]).shape)
#print(ca1)

sa1=m1.reshape(51,36)                           #纬向速度
za1=n1.reshape(51,36)                           #位势高度场
qa1=np.dot(np.transpose(m1),sa)                 #时间系数Q  (1,1836)*(1836,40)
ra1=np.dot(np.transpose(n1),za)                 #时间系数R  (1,1836)*(1836,40)

#uw1=m1.reshape(51,36)                           #纬向速度
#zw1=n1.reshape(51,36)                           #位势高度场
#qw1=np.dot(np.transpose(m1),sw)                 #时间系数Q  (1,1836)*(1836,40)
#rw1=np.dot(np.transpose(n1),zw)                 #时间系数R  (1,1836)*(1836,40)

#第二模态
ua2=m2.reshape(51,36)                           #纬向速度
za2=n2.reshape(51,36)                           #位势高度场
qa2=np.dot(np.transpose(m2),ua)                 #时间系数Q  (1,1836)*(1836,40)
ra2=np.dot(np.transpose(n2),za)                 #时间系数R  (1,1836)*(1836,40)

#uw2=m2.reshape(51,36)                           #纬向速度
#zw2=n2.reshape(51,36)                           #位势高度场
#qw2=np.dot(np.transpose(m2),sw)                 #时间系数Q  (1,1836)*(1836,40)
#rw2=np.dot(np.transpose(n2),zw)                 #时间系数R  (1,1836)*(1836,40)

#第三模态
sa3=m3.reshape(51,36)                           #纬向速度
za3=n3.reshape(51,36)                           #位势高度场
qa3=np.dot(np.transpose(m3),sa)                 #时间系数Q  (1,1836)*(1836,40)
ra3=np.dot(np.transpose(n3),za)                 #时间系数R  (1,1836)*(1836,40)

#第四模态
sa4=m4.reshape(51,36)                           #纬向速度
za4=n4.reshape(51,36)                           #位势高度场
qa4=np.dot(np.transpose(m4),sa)                 #时间系数Q  (1,1836)*(1836,40)
ra4=np.dot(np.transpose(n4),za)                 #时间系数R  (1,1836)*(1836,40)


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


#解决0度经线出现白条
from cartopy.util import add_cyclic_point
#ua1,lon=add_cyclic_point(ua1,coord=lon)
za2,lon=add_cyclic_point(za2,coord=lon)
#ua2,lon=add_cyclic_point(ua2,coord=lon)
#za3,lon=add_cyclic_point(za3,coord=lon)
#sa3,lon=add_cyclic_point(sa3,coord=lon)
#za4,lon=add_cyclic_point(za4,coord=lon)
#sa4,lon=add_cyclic_point(sa4,coord=lon)


import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueDarkRed18
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

plt.contourf(lon, lat, za2,cmap=cp)
#MPL_PiYG_r\GMT_polar\BlueDarkRed18\GMT_panoply
plt.title('SVD_ZA2_Z500',loc='center',fontsize=15,pad=15)
#D-val  High   Low

#,levels=np.arange(19620,20269,36)  levels=np.linspace(4880,5600,10),
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=ax, orientation="vertical", norm='normlize',pad=0.07, aspect=40, shrink=1)
cb.set_label('U',size=3,rotation=0,labelpad=5,fontsize=12)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=10)
print("hello,world!")
x=np.arange(1979,2019)
plt.savefig('/home/dell/ZQ19/SVD_ZA2_Z500.jpg')
#plt.show()

#print(ua)
#print(qa1)
#print(ra1)
qa2=qa2.reshape(40,1)
ra2=ra2.reshape(40,1)
fig,ax1=plt.subplots()
ax1.plot(x,qa2,c='r')
ax1.plot(x,ra2,c='b')
plt.savefig('/home/dell/ZQ19/SVD_TA2_Z500.jpg')

