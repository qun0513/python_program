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

#去趋势海冰指数
bw=[-0.00361826 , 0.0210011  , 0.06757824 ,-0.07504757 ,-0.06303819 ,-0.09119824,
  0.02557659 , 0.01648379 , 0.00853077 , 0.05870681 ,-0.05212771 ,-0.05121047,
 -0.05458098, -0.04672296 , 0.03658021 ,-0.04101541 , 0.01179921 , 0.06191167,
  0.1136747 ,  0.12019503, -0.00768342 ,-0.02620504 , 0.02010535 , 0.11745327,
  0.11492807 ,-0.03507578, -0.09117778, -0.05772816, -0.08261504 , 0.00737763,
  0.03909296 , 0.13668505 ,-0.01027709, -0.06943627 ,-0.00165718 , 0.09946517,
 -0.07199829 ,-0.15846179, -0.09349538, -0.00214966]#,  0.10937505]
ew=[ 0.07748456, -0.00074116,  0.00097511,  0.05744472, -0.06073609 , 0.04067163,
  0.00090103,  0.02097827, -0.00604557, -0.04224639, -0.0057965 , -0.08954526,
 -0.01942958 , 0.02865336 , 0.03420822,  0.03765049 ,-0.09566951 ,-0.0455377,
  0.0005391 ,  0.03331345, -0.0314    ,  0.09198162,  0.03148051 , 0.04246037,
 -0.00829156, -0.02204904 ,-0.06304809 , 0.00598037 ,-0.02863671 ,-0.08426499,
 -0.02284624 ,-0.0325508  , 0.06105801 ,-0.00312563 ,-0.02610089 ,-0.01318278,
  0.02378648 ,-0.02411578, -0.02234231 , 0.08259528]#  ,0.07554004]
ba=[-0.08475741,  0.0294586 , -0.03016749 , 0.07275441, -0.03833596 ,-0.11866105,
 -0.05009551, -0.00855759 , 0.03904933,  0.09161654,  0.02786033 ,-0.04129688,
  0.01265778 , 0.03811367 , 0.05430281,  0.03757353, -0.01158646, -0.05077826,
  0.02703395,  0.08743253 , 0.01064093 ,-0.04862271 ,-0.03134865,  0.06462187,
  0.06150678, -0.01848365 , 0.00975299,  0.00818998 ,-0.0575875 ,  0.01298803,
 -0.01327381 , 0.03546445, -0.00438902 ,-0.07188649 ,-0.05914233 , 0.12563519,
 -0.02971634 ,-0.04702351 ,-0.03355168 ,-0.0520554]# ,  0.05466495]
ea=[ 4.2701289e-03 , 3.1950464e-03, -9.4881700e-04  ,9.8816631e-04,
  1.4278758e-03 , 1.0684496e-03 , 8.7881554e-04, -8.7412726e-04,
  1.6075978e-04 ,-1.0763784e-03 ,-1.8739421e-04 ,-3.4241937e-04,
 -2.4567044e-03 , 2.6696082e-04,  6.1021559e-04 , 1.7018300e-03,
 -2.2582319e-03 ,-1.7856343e-03 ,-4.5558815e-03 , 2.3342599e-03,
 -4.4344794e-03 , 1.8273094e-03 ,-1.6104241e-03  ,1.0352228e-03,
 -2.4507325e-03 ,-1.4272698e-03 ,-3.2147970e-03 ,-7.0258789e-04,
 -2.6803515e-03 ,-3.8881686e-03 ,-4.6046427e-04 ,-3.9976300e-04,
  7.2294683e-04 ,-2.7836091e-04 , 3.2907440e-03 , 3.4659961e-03,
  4.5894654e-03  ,3.6402124e-03 , 3.2617128e-04  ,8.5846288e-05]#,1.4654733e-04]

#保留趋势  1
bw1=[0.54868827, 0.5684517,  0.61017292, 0.46269118, 0.46984463, 0.43682865,
 0.54874756, 0.53479883, 0.52198988, 0.56731   , 0.45161955,0.44768086,
 0.43945443, 0.44245651, 0.52090377, 0.43845221, 0.48641091, 0.53166744,
 0.57857454, 0.58023895, 0.44750457, 0.42412702, 0.46558149, 0.55807348,
 0.55069235, 0.39583257, 0.33487465, 0.36346834, 0.33372553, 0.41886227,
 0.44572168, 0.53845784, 0.38663977, 0.32262467, 0.38554783, 0.48181426,
 0.30549486, 0.21417544, 0.27428592, 0.36077571]#, 0.4674445 ]
ew1=[0.35071743, 0.27118743, 0.27159943, 0.32676476, 0.20727967, 0.30738311,
 0.26630823, 0.28508119, 0.25675307, 0.21924797, 0.25439358, 0.16934054,
 0.23815195, 0.28493061, 0.28918118, 0.29131918, 0.1566949 , 0.20552242,
 0.25029495, 0.28176501, 0.21574729, 0.33782463, 0.27601924, 0.28569482,
 0.23363861, 0.21857685, 0.17627352, 0.2439977 , 0.20807635, 0.15114378,
 0.21125825, 0.20024941, 0.29255395, 0.22706603, 0.20278649, 0.21440032,
 0.2500653 , 0.20085877, 0.20132795, 0.30496126]#, 0.29660174]
ba1=[0.11516363, 0.22724025, 0.16547479, 0.26625732, 0.15302755, 0.07056309,
 0.13698925, 0.17638779, 0.22185533, 0.27228317, 0.20638756, 0.13509098,
 0.18690626, 0.21022277, 0.22427253, 0.20540388, 0.1541045,  0.11277332,
 0.18844616, 0.24670535, 0.16777438, 0.10637137, 0.12150604, 0.21533717,
 0.21008271, 0.12795289, 0.15405016, 0.15034777, 0.0824309,  0.15086706,
 0.12246585, 0.16906472, 0.12707187, 0.05743502, 0.0680398,  0.25067794,
 0.09318703, 0.07374048, 0.08507293, 0.06442983]#, 0.1690108 ]
ea1=[0.00991863, 0.00878285, 0.00457828, 0.00645457, 0.00683357, 0.00641345,
 0.00616311,0.00434947, 0.00532365, 0.00402581, 0.0048541,  0.00463837,
 0.00246338, 0.00512635, 0.0054089 , 0.00643981, 0.00241905, 0.00283095,
 0.        , 0.00682944, 0.        , 0.00620109, 0.00270265, 0.0052876,
 0.00174094, 0.0027037 , 0.00085548, 0.00330698, 0.00126852, 0.       ,
 0.003367  , 0.003367  , 0.00442901, 0.003367  , 0.00687541, 0.00698996,
 0.00805273, 0.00704277, 0.00366803, 0.003367  ]#, 0.003367  ]

gph=xr.open_dataset('/media/usb/Reanalysis/ECMWF/Interim/monthly/Z_1979-201908.nc')
gph=gph/10

#print(gph)
g=gph.z.loc[:,50,90:40,0:359]                                #数据level
lon=g.longitude
lat=g.latitude

#位势高度场
#秋季
year=np.arange(1979,2019)
t1=g.sel(time=g.time.dt.month.isin([9]))
t2=g.sel(time=g.time.dt.month.isin([10]))
t3=g.sel(time=g.time.dt.month.isin([11]))
T1=t1.sel(time=t1.time.dt.year.isin([year]))
T2=t2.sel(time=t2.time.dt.year.isin([year]))
T3=t3.sel(time=t3.time.dt.year.isin([year]))
TA=(T1.data+T2.data+T3.data)/3
#冬季
year1=np.arange(1979,2019)
year2=np.arange(1980,2020)
t4=g.sel(time=g.time.dt.month.isin([12]))
t5=g.sel(time=g.time.dt.month.isin([1]))
t6=g.sel(time=g.time.dt.month.isin([2]))
t7=g.sel(time=g.time.dt.month.isin([3]))
t8=g.sel(time=g.time.dt.month.isin([4]))
t9=g.sel(time=g.time.dt.month.isin([5]))
T4=t4.sel(time=t4.time.dt.year.isin([year1]))
T5=t5.sel(time=t5.time.dt.year.isin([year2]))
T6=t6.sel(time=t6.time.dt.year.isin([year2]))
T7=t7.sel(time=t7.time.dt.year.isin([year2]))
T8=t8.sel(time=t8.time.dt.year.isin([year2]))
T9=t9.sel(time=t9.time.dt.year.isin([year2]))

TW=(T4.data+T5.data+T6.data)/3

#滞后性
TA1=(T2.data+T3.data+T4.data)/3  #秋+1
TA2=(T3.data+T4.data+T5.data)/3  #秋+2
TA3=(T4.data+T5.data+T6.data)/3  #秋+3

TW1=(T5.data+T6.data+T7.data)/3  #冬+1
TW2=(T6.data+T7.data+T8.data)/3  #冬+2
TW3=(T7.data+T8.data+T9.data)/3  #冬+3



#T=(T1+T2+T3)/3
#print(T1)
#print(T1.data)
import scipy
from scipy import signal
#T=scipy.signal.detrend(T)                           #去趋势

import pandas as pd
bw1=pd.Series(bw1)                        #海冰指数  保留趋势
ew1=pd.Series(ew1)
ba1=pd.Series(ba1)
ea1=pd.Series(ea1)
bw=pd.Series(bw)                          #海冰指数  去趋势
ew=pd.Series(ew)
ba=pd.Series(ba)
ea=pd.Series(ea)

rbw=np.empty((51,360));pbw=np.empty((51,360))
rew=np.empty((51,360));pew=np.empty((51,360))
rba=np.empty((51,360));pba=np.empty((51,360))
rea=np.empty((51,360));pea=np.empty((51,360))

rbw1=np.empty((51,360));pbw1=np.empty((51,360))
rew1=np.empty((51,360));pew1=np.empty((51,360))
rba1=np.empty((51,360));pba1=np.empty((51,360))
rea1=np.empty((51,360));pea1=np.empty((51,360))

rbw2=np.empty((51,360));pbw2=np.empty((51,360))
rew2=np.empty((51,360));pew2=np.empty((51,360))
rba2=np.empty((51,360));pba2=np.empty((51,360))
rea2=np.empty((51,360));pea2=np.empty((51,360))

rbw3=np.empty((51,360));pbw3=np.empty((51,360))
rew3=np.empty((51,360));pew3=np.empty((51,360))
rba3=np.empty((51,360));pba3=np.empty((51,360))
rea3=np.empty((51,360));pea3=np.empty((51,360))
#相关性分析+显著性检验--------------------------------------------------------
from scipy.stats import pearsonr
for i in np.arange(0,51):
    for j in np.arange(0,360):
       w=pd.Series(TW[:,i,j])           #冬季 位势高度场
       w1=pd.Series(TW1[:,i,j])
       w2=pd.Series(TW2[:,i,j])
       w3=pd.Series(TW3[:,i,j])

       a=pd.Series(TA[:,i,j])           #秋季 位势高度场
       a1=pd.Series(TA1[:,i,j])
       a2=pd.Series(TA2[:,i,j])
       a3=pd.Series(TA3[:,i,j])
       #rbw[i,j]=w.corr(bw1,method='pearson')
       #rew[i,j]=w.corr(ew1,method='pearson')
       #rba[i,j]=a.corr(ba1,method='pearson')
       #rea[i,j]=a.corr(ea1,method='pearson')

       #滞后性-----------------------------------

       rbw[i,j],pbw[i,j]=pearsonr(bw1,w) #滞后0
       rew[i,j],pew[i,j]=pearsonr(ew1,w)
       rba[i,j],pba[i,j]=pearsonr(ba,a)
       rea[i,j],pea[i,j]=pearsonr(ea1,a)

       #rbw1[i,j],pbw1[i,j]=pearsonr(bw,w1) #滞后1
       #rew1[i,j],pew1[i,j]=pearsonr(ew,w1)
       rba1[i,j],pba1[i,j]=pearsonr(ba1,a1)
       #rea1[i,j],pea1[i,j]=pearsonr(ea,a1)

       #rbw2[i,j],pbw2[i,j]=pearsonr(bw,w2) #滞后2
       #rew2[i,j],pew2[i,j]=pearsonr(ew,w2)
       rba2[i,j],pba2[i,j]=pearsonr(ba1,a2)
       #rea2[i,j],pea2[i,j]=pearsonr(ea,a2)

       #rbw3[i,j],pbw3[i,j]=pearsonr(bw,w3) #滞后3
       #rew3[i,j],pew3[i,j]=pearsonr(ew,w3)
       rba3[i,j],pba3[i,j]=pearsonr(ba1,a3)
       #rea3[i,j],pea3[i,j]=pearsonr(ea,a3)
#print(r)
#print(r.shape)
print(lat.size)
print(lon.size)
#print(pbw)

#area1=np.where(pbw<0.05)
#area2=np.where(pew<0.05)
#area3=np.where(pba<0.05)
#area4=np.where(pea<0.05)

#area11=np.where(pbw1<0.05)
#area21=np.where(pew1<0.05)
#area31=np.where(pba1<0.05)
#area41=np.where(pea1<0.05)

#area12=np.where(pbw2<0.05)
#area22=np.where(pew2<0.05)
#area32=np.where(pba2<0.05)
#area42=np.where(pea2<0.05)

#area13=np.where(pbw3<0.05)
#area23=np.where(pew3<0.05)
#area33=np.where(pba3<0.05)
area43=np.where(pea3<0.05)
#area1=area1[::3]
#area2=area2[::3]
#area3=area3[::3]
#area4=area4[::3]
#print(area1)





#绘图--------------------------------------------------------------
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
           , labelsize=7  # ,thetalabels=np.arange(0,360,45))
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
#相关系数
#rbw3,lon=add_cyclic_point(rbw3,coord=lon)
#rew3,lon=add_cyclic_point(rew3,coord=lon)
rbw,lon=add_cyclic_point(rbw,coord=lon)
#rea3,lon=add_cyclic_point(rea,coord=lon)
#4*4*2
#显著性检验方法1------
#nx,ny=np.meshgrid(lon,lat)
#xx=nx[area43]
#yy=ny[area43]

import cmaps
from matplotlib.colors import ListedColormap
cmap1=cmaps.BlueDarkRed18
list_cmap1=cmap1(np.linspace(0,1,256))
cp=ListedColormap(list_cmap1,name='cp')

rr=ax.contourf(lon,lat, rbw,levels=np.linspace(-0.75,0.75,11),cmap=cp,zorder=0)  #MPL_PiYG_r
plt.title('ST_Bs_Winter_50',loc='center',fontsize=23,pad=12)##ST-Significance test
cb = plt.colorbar(rr, orientation="vertical", pad=0.00007, aspect=20, shrink=1)
cb.set_label('r',size=5,rotation=0,labelpad=5,fontsize=15)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=12)
#打点-------
#pp=ax.scatter(xx[::7],yy[::7],marker='.',s=2.8,c='k',alpha=1)
import numpy.ma as ma
pbw=ma.MaskedArray(data = pbw, mask = np.logical_and(pbw<1,pbw>0.05))
lon=g.longitude
lon,lat=np.meshgrid(lon,lat)

pp=ax.contourf(lon,lat,pbw,hatches=['...'],colors="none",zorder=0)

#,levels=np.arange(19620,20269,36)  levels=np.linspace(4880,5600,10),
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

print("hello,world!")
#plt.savefig('/home/dell/ZQ19/ST_Os_AA_50.jpg')
plt.savefig('/home/dell/ZQ19/rrr/ST_Bs_Win_50.jpg')
#plt.show()
print('Hello,world!')



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

'''
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import proplot as plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math

fig,ax= plt.subplots(figsize=(10,6),dpi=98)
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([0,180,50,90])
ax.coastlines(resolution="50m",linewidth=1)  #resolution:110m\ 50m\ 10m

import cmaps
plt.contourf(lon,lat, rea,levels=np.linspace(-0.60,0.45,8),transform=ccrs.PlateCarree(),cmap=cmaps.BlueDarkRed18)  #MPL_PiYG_r
plt.title('r_Os_A_50',position=(0.5,0.9),loc='center',fontsize=23,pad=12)

#,levels=np.arange(19620,20269,36)  levels=np.linspace(4880,5600,10),
#cmap分类：连续类、分色类、循环类、定性类、混杂类。

cb = plt.colorbar(ax=ax, orientation="horizontal", pad=0.07, aspect=40, shrink=1)
cb.set_label('r',size=3,rotation=0,labelpad=5,fontsize=15)
#cb.set_xticks([])
cb.ax.tick_params(labelsize=12)
'''