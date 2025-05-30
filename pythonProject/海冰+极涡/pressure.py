import xarray as xr
import matplotlib             #解决Linux无法可视化的问题
matplotlib.use('Agg')         #
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from xarray import DataArray

gph=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/Z_1979-201908.nc')
#print(gph)
gph=gph/10

#bs=p.z.loc[:,1000,82.5:66.5,15.5:60.5]    #巴伦支海海域     1000 level
#es=p.z.loc[:,1000,62.5:44.5,135.5:164.5]  #鄂霍茨克海海域 的 1000 level
g=gph.z.loc[:,850,90:40,0:359]
#print(g)
lon=g.longitude
lat=g.latitude
g_bar=g.mean(dim='time')

#t[0,3]  T[0,7]   dif[1,8]---------------

#巴伦支海 秋季 高低年份---------------------------------------------------------------
year0=[1982,1988,1993,1998,2002,2003,2014,2019]          #bs high
year1=[1979,1984,2007,2012,2013]                         #bs low
t0=g.sel(time=g.time.dt.month.isin([9,10,11]))           #秋季
T0=t0.sel(time=t0.time.dt.year.isin([year0]))
T1=t0.sel(time=t0.time.dt.year.isin([year1]))
#b_Autumn=np.array(T0).reshape(41,3)                     #巴伦支海秋季  high 三个月份
bha=T0.mean(dim='time')                                  #high 数据
bla=T1.mean(dim='time')                                  #low  数据
#print(bha)

t01=g.sel(time=g.time.dt.month.isin([9]))
t02=g.sel(time=g.time.dt.month.isin([10]))
t03=g.sel(time=g.time.dt.month.isin([11]))
T01=t01.sel(time=t01.time.dt.year.isin([year0]))    #high
T02=t02.sel(time=t02.time.dt.year.isin([year0]))
T03=t03.sel(time=t03.time.dt.year.isin([year0]))
T11=t01.sel(time=t01.time.dt.year.isin([year1]))    #low
T12=t02.sel(time=t02.time.dt.year.isin([year1]))
T13=t03.sel(time=t03.time.dt.year.isin([year1]))

b_ha=(T01.data+T02.data+T03.data)/3   #high(7)
b_la=(T11.data+T12.data+T13.data)/3   #low (8)

'''
dif11=np.empty((7,51,360))
dif11_bar=bla-bha
s=0
for k in np.arange(0,7):
    dif11[k,:,:]=(b_ha[k,:,:]-g_bar)
    s=s+dif11[k,:,:]
dif11_bar=s/7                           
'''

dif1=np.empty((7,51,360))
dif2=np.empty((5,51,360))
s=0
for k in np.arange(0,7):
    dif1[k,:,:]=(b_ha[k,:,:]-g_bar)
    s=s+dif1[k,:,:]
dif1_bar=s/7                           #(os high — D-val) mean
s=0
for k in np.arange(0,5):
    dif2[k,:,:]=(b_la[k,:,:]-g_bar)    #看位势高度场升高
    s=s+dif2[k,:,:]
dif2_bar=s/5                           #(os low — D-val) mean


#巴伦支海 冬季 高低年份-----------------------------------------------------------
year2=[1997,1998,2002,2003,2010,2014,2019]               #bs high
year_2=[1998,1999,2003,2004,2011,2015,2020]
year3=[1982,1984,2005,2007,2015,2016,2017]               #bs low
year_3=[1983,1985,2006,2008,2016,2017,2018]
t1=g.sel(time=g.time.dt.month.isin([12]))                #冬季
t_1=g.sel(time=g.time.dt.month.isin([1,2]))
T2=t1.sel(time=t1.time.dt.year.isin([year2]))
T_2=t_1.sel(time=t_1.time.dt.year.isin([year_2]))
T2=T2.mean(dim='time')
T_2=T_2.mean(dim='time')
bhw=T2.data/3+T_2.data*2/3                                #high
T3=t1.sel(time=t1.time.dt.year.isin([year3]))
T_3=t_1.sel(time=t_1.time.dt.year.isin([year_3]))
T3=T3.mean(dim='time')
T_3=T_3.mean(dim='time')
blw=T3.data/3+T_3.data*2/3                                #low

t11=g.sel(time=g.time.dt.month.isin([12]))
t12=g.sel(time=g.time.dt.month.isin([1]))
t13=g.sel(time=g.time.dt.month.isin([2]))

T21=t11.sel(time=t11.time.dt.year.isin([year2]))   #high
T22=t12.sel(time=t12.time.dt.year.isin([year_2]))
T23=t13.sel(time=t13.time.dt.year.isin([year_2]))

T31=t11.sel(time=t11.time.dt.year.isin([year3]))   #low
T32=t12.sel(time=t12.time.dt.year.isin([year_3]))
T33=t13.sel(time=t13.time.dt.year.isin([year_3]))

b_hw=(T21.data+T22.data+T23.data)/3  #high(6)
b_lw=(T31.data+T32.data+T33.data)/3  #low(5)

dif3=np.empty((6,51,360))
dif4=np.empty((7,51,360))
s=0
for k in np.arange(0,6):
    dif3[k,:,:]=(b_hw[k,:,:]-g_bar)
    s=s+dif3[k,:,:]
dif3_bar=s/6                           #(os high — D-val) mean (高值年逐年差值求和后平均)
s=0
for k in np.arange(0,7):
    dif4[k,:,:]=(b_lw[k,:,:]-g_bar)
    s=s+dif4[k,:,:]
dif4_bar=s/7                           #(os low — D-val) mean


#鄂霍茨克海 秋季 高低值年份----------------------------------------------------------
year4=[1979,1980,1998,2013,2014,2015,2016]
year5=[1991,1995,1997,1999,2003,2005,2007,2008]
t2=g.sel(time=g.time.dt.month.isin([9,10,11]))
T4=t2.sel(time=t2.time.dt.year.isin([year4]))
T5=t2.sel(time=t2.time.dt.year.isin([year5]))
oha=T4.mean(dim='time')                                  #high 数据
ola=T5.mean(dim='time')                                  #low  数据

t21=g.sel(time=g.time.dt.month.isin([9]))
t22=g.sel(time=g.time.dt.month.isin([10]))
t23=g.sel(time=g.time.dt.month.isin([11]))
T41=t21.sel(time=t21.time.dt.year.isin([year4]))    #high
T42=t22.sel(time=t22.time.dt.year.isin([year4]))
T43=t23.sel(time=t23.time.dt.year.isin([year4]))
T51=t21.sel(time=t21.time.dt.year.isin([year5]))    #low
T52=t22.sel(time=t22.time.dt.year.isin([year5]))
T53=t23.sel(time=t23.time.dt.year.isin([year5]))

o_ha=(T41.data+T42.data+T43.data)/3   #high(7)
o_la=(T51.data+T52.data+T53.data)/3   #low (8)

dif5=np.empty((7,51,360))
dif6=np.empty((8,51,360))
s=0
for k in np.arange(0,7):
    dif5[k,:,:]=(o_ha[k,:,:]-g_bar)
    s=s+dif5[k,:,:]
dif5_bar=s/7                           #(os high — D-val) mean
s=0
for k in np.arange(0,8):
    dif6[k,:,:]=(o_la[k,:,:]-g_bar)
    s=s+dif6[k,:,:]
dif6_bar=s/8                           #(os low — D-val) mean

#鄂霍茨克海 冬季 高低年份---------------------------------------------------------
year6=[1979,1982,2000,2011,2018,2019]                    #es high
year_6=[1980,1983,2001,2012,2019,2020]
year7=[1983,1990,1995,2005,2008]                         #es low
year_7=[1984,1991,1996,2006,2009]
t3=g.sel(time=g.time.dt.month.isin([12]))                #冬季
t_3=g.sel(time=g.time.dt.month.isin([1,2]))
T6=t3.sel(time=t3.time.dt.year.isin([year6]))
T_6=t_3.sel(time=t_3.time.dt.year.isin([year_6]))
T6=T6.mean(dim='time')
T_6=T_6.mean(dim='time')
ohw=T6.data/3+T_6.data*2/3                                #high
T7=t3.sel(time=t3.time.dt.year.isin([year7]))
T_7=t_3.sel(time=t_3.time.dt.year.isin([year_7]))
T7=T7.mean(dim='time')
T_7=T_7.mean(dim='time')
olw=T7.data/3+T_7.data*2/3                                #low

t31=g.sel(time=g.time.dt.month.isin([12]))
t32=g.sel(time=g.time.dt.month.isin([1]))
t33=g.sel(time=g.time.dt.month.isin([2]))

T61=t31.sel(time=t31.time.dt.year.isin([year6]))   #high
T62=t32.sel(time=t32.time.dt.year.isin([year_6]))
T63=t33.sel(time=t33.time.dt.year.isin([year_6]))

T71=t31.sel(time=t31.time.dt.year.isin([year7]))   #low
T72=t32.sel(time=t32.time.dt.year.isin([year_7]))
T73=t33.sel(time=t33.time.dt.year.isin([year_7]))

o_hw=(T61.data+T62.data+T63.data)/3  #high(6)
o_lw=(T71.data+T72.data+T73.data)/3  #low(5)

dif7=np.empty((5,51,360))
dif8=np.empty((5,51,360))
s=0
for k in np.arange(0,5):
    dif7[k,:,:]=(o_hw[k,:,:]-g_bar)
    s=s+dif7[k,:,:]
dif7_bar=s/5                           #(os high — D-val) mean (高值年逐年差值求和后平均)
s=0
for k in np.arange(0,5):
    dif8[k,:,:]=(o_lw[k,:,:]-g_bar)
    s=s+dif8[k,:,:]
dif8_bar=s/5                           #(os low — D-val) mean




import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import proplot as plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math



proj = plot.Proj('npstere', central_longitude=90)
fig, ax = plot.subplots(proj=proj)
ax.format(labels=False, grid=False, coast=True, metalinewidth=1.0, boundinglat=40)
#metalinewidth 外边框、 boundlinglat 最外围纬圈

#gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1, color='lightgrey', x_inline=True, y_inline=True,
#                  xlocs=np.arange(-180, 180,45)) #(-180, 180,45) 0,360, 45

ins = ax.inset_axes([0., 0., 1, 1], proj='polar', zorder=0)
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
           , labelsize=50  # ,thetalabels=np.arange(0,360,45))
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

'''高低值年-------------------------（有用）

#解决0度经线出现白条
from cartopy.util import add_cyclic_point
#bha,lon=add_cyclic_point(bha,coord=lon)     #巴伦支海
#bla,lon=add_cyclic_point(bla,coord=lon)
#bhw,lon=add_cyclic_point(bhw,coord=lon)
#blw,lon=add_cyclic_point(blw,coord=lon)

#oha,lon=add_cyclic_point(oha,coord=lon)     #鄂霍茨克海
#ola,lon=add_cyclic_point(ola,coord=lon)
#ohw,lon=add_cyclic_point(ohw,coord=lon)
#olw,lon=add_cyclic_point(olw,coord=lon)

#差值
#dif1=bla-bha                                 #巴伦支海秋季
#dif2=blw-bhw
#dif3=ola-oha
#dif4=olw-ohw

#dif1,lon=add_cyclic_point(dif1,coord=lon)
'''

#单样本t检验
from scipy.stats import ttest_1samp
#t1,p1=ttest_1samp(dif1,0)
#t2,p2=ttest_1samp(dif2,0)
#t3,p3=ttest_1samp(dif3,0)
#t4,p4=ttest_1samp(dif4,0)
#t5,p5=ttest_1samp(dif5,0)
#t6,p6=ttest_1samp(dif6,0)
#t7,p7=ttest_1samp(dif7,0)
t8,p8=ttest_1samp(dif8,0)
 #差值dif1_bar

#双样本t检验
from scipy import stats
from scipy.stats import ttest_ind
t11,p11=ttest_ind(b_ha,b_la,equal_var=False)
#t22,p22=ttest_ind(b_hw,b_lw,equal_var=False)
#t33,p33=ttest_ind(o_ha,o_la,equal_var=False)
t44,p44=ttest_ind(o_hw,o_lw,equal_var=False)
 #差值
#dif11=bla-bha                                 #巴伦支海秋季
#dif22=blw-bhw
#dif33=ola-oha
dif44=olw-ohw

print(t11,t11.shape,p11.min())
print(dif5.shape)
print(t8.shape)
print(p8.shape)

#打点区域(scatter)------------
area5=np.where(p8<0.05)
nx,ny=np.meshgrid(lon,lat)
xx=nx[area5]
yy=ny[area5]

#掩膜掉不打点的区域
import numpy.ma as ma
p44=ma.MaskedArray(data = p44, mask = np.logical_and(p44<1,p44>0.05))
#解决0度经线出现白条
from cartopy.util import add_cyclic_point
dif44,lon=add_cyclic_point(dif44,coord=lon)

#绘图--------------------------------------------------------------------------
import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueDarkRed18  #BlueDarkRed18  GMT_panoply
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

rr=ax.contourf(lon,lat, dif44,levels=np.linspace(-90,90,10),extend='both',cmap=cp,zorder=0)  #MPL_PiYG_r
#,levels=np.arange(-50,50,11)
plt.title('ST(t2)_Os_D-val_Win_Z850',loc='center',fontsize=12,pad=12)  ##ST-Significance test
cb = plt.colorbar(rr, orientation="vertical", pad=0.00007, aspect=20, shrink=1)
cb.set_label('',size=5,rotation=0,labelpad=5,fontsize=15)
#cb.set_xticks([])--------------
cb.ax.tick_params(labelsize=12)

#打点---------------------------------------------
#pp=ax.scatter(xx[::7],yy[::7],marker='.',s=2.8,c='k',alpha=1) #-----------
lon=g.longitude
pp=ax.contourf(lon,lat,p44,hatches=['....'],colors="none",zorder=0)
#,levels=np.arange(19620,20269,36)  levels=np.linspace(4880,5600,10),
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

print("hello,world!")
plt.savefig('/home/dell/ZQ19/p20230124.jpg')
#plt.show()


'''高低值年---------------------------------（有用）
cpr=bha.max()
cpl=bha.min()
print(cpl,cpr)
import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueDarkRed18
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

plt.contourf(lon, lat, dif,cmap=cp)
#MPL_PiYG_r\GMT_polar\BlueDarkRed18\GMT_panoply
plt.title('Os_Win_D-val_Z50',loc='center',fontsize=20,pad=15)
#D-val  High   Low

#,levels=np.arange(19620,20269,36)  levels=np.linspace(4880,5600,10),
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=ax, orientation="vertical", norm='normlize',pad=0.07, aspect=40, shrink=1)
cb.set_label('Z',size=3,rotation=0,labelpad=5,fontsize=12)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=10)
print("hello,world!")
plt.savefig('/home/dell/ZQ19/Os_Win_D-val_Z50.jpg')
#plt.show()
'''



'''
fig,ax= plt.subplots(figsize=(10,6),dpi=98)
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([0,180,50,90])
ax.coastlines(resolution="50m",linewidth=1)  #resolution:110m\ 50m\ 10m

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')
gl.top_labels=False    #gl.xlabels_top = False
gl.right_labels=False  #gl.ylabels_right = False
gl.xlines = False
gl.ylines=False
#gl.xlocator = mticker.FixedLocator([-30,15,60,75,120])
#gl.ylocator = mticker.FixedLocator([0,22.5,45,67.5])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':18, 'color':'black'}
gl.ylabel_style = {'size':18, 'color':'black'}
'''