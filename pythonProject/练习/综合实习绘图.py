from cartopy.io.shapereader import Reader
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.path as mpath
from matplotlib.path import  Path
plt.rcParams['font.sans-serif']=['KaiTi'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

'''
#常规观测
t1=[67.5,67.5,68,68,68,68,68]  #地表最高温度
t2=[10.3,10.3,10.3,10.3,10,10,10]  #地表最低温度
t3=[45,50.5,60,62,59,49.2,37.5]  #地表温度
t4=[26.5,28.3,30,32.5,34,34,34]  #空气最高温度
t5=[22.2,22.2,22.2,22.2,22,22,20]  #空气最低温度
t6=[17.4,16.3,17.8,17,18.2,18.4,18]  #湿球温度
t7=[25.8,25.9,29.4,31.3,33.2,33.4,32]  #干球温度

s1=[17.9,16.9,18.5,17.9,19.1,19.3,18.9]  #订正后湿球温度
s2=[15.2,13.2,14.1,11.7,12.8,13.1,13.0]  #水汽压
s3=[46,40,34.2,25.3,25,25.4,27.5]  #相对湿度
s4=[13.3,11.1,12.1,9.2,10.6,10.93,10.85]  #露点温度

p1=[823,822.4,822.5,821.6,821.2,821.3,820.4]  #观测气压
p2=[818.57,817.92,818.03,817.13,816.55,816.75,815.75]  #订正后气压
p3=[1000.05,998.93,999.06,998.03,994,994.48,993.14]  #海平面气压

#大的自动站
q1=[27,30.1,31.4,35,36.09,35.7,35.9]  #温度
q2=[40.3,34.59,28.9,23.5,28.88,25.8,20.1]  #湿度
q3=[817.2,816.9,816.5,815.8,815.1,816.6,813.8]  #气压
q4=[1.19,0.66,1.81,1.25,1.59,0,0.93]  #风速
q5=[] #风向

#手持式自动气象站
longitude=104.09
latitude=35.57
r1=[28.8,31.4,31.2,32.4,33.5,37,34]  #温度
r2=[40.5,34.8,29.7,27.6,23.8,20.2,28.3]  #湿度
r3=[819.2,818.8,818.5,817.5,817.1,814.6,815.7]  #气压
r4=[0.6,1,0.8,1.2,1.2,0.44,1.5]  #风速
r5=[962.7,962.5,962.2,961.1,960.4,959.8,958.8]  #海平面气压
r6=[1756,1755,1761,1769,1777,1777,1790]  #海拔高度

fig,ax=plt.subplots()
x=[0,1,2,3,4,5,6]
ax.set_title('常规观测--other1',fontsize=18)
ax.plot(x,p1,c='darkviolet',label='观测气压',ls='-',marker='*')
ax.plot(x,p2,c='orange',label='订正后气压',ls='-.',marker='o')

ax.tick_params(#top=True,bottom=True,left=True,right=True,
               width=1.5,direction='in',length=6,labelsize=15)
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xticklabels(['9:30','10:30','11:30','12:30','13:30','14:30','15:30'])
ax.set_xlabel('时间',fontsize=15)
ax.set_ylabel('要素值',fontsize=15)
plt.legend(loc='lower left',fontsize=12)
ax2=ax.twinx()
ax2.plot(x,p3,c='deepskyblue',label='海平面气压',ls='--',marker='^')
ax.plot(x,p1,c='darkviolet',label='观测气压',ls='-',marker='*')
ax.plot(x,p2,c='orange',label='订正后气压',ls='-.',marker='o')
ax2.tick_params(#top=True,bottom=True,left=True,right=True,
               width=1.5,direction='in',length=6,labelsize=15)
ax2.set_xticks([0,1,2,3,4,5,6])
ax2.set_xticklabels(['9:30','10:30','11:30','12:30','13:30','14:30','15:30'])
ax2.set_xlabel('时间',fontsize=15)
ax2.set_ylabel('要素值',fontsize=15)
plt.legend(loc='best',fontsize=12)
plt.show()
'''
'''
#野外其他数据
x=[0,1,2,3,4]
#x=['兰州','白银','乌鞘岭','武威','张掖']
y1=[1755,1680,2910,1544,1462] #海拔
y2=[30.1,25.7,22.3,30.9,30.5] #温度
y3=[34.59,55,34.8,59.7,38]    #湿度
y4=[816.9,826.8,708.9,841.4,848.5] #气压
y5=[962.5,965.3,833.4,989.2,997.1] #海平面气压
fig,ax=plt.subplots()
ax.plot(x,y2,c='deepskyblue',label='温度',ls='-',marker='o')
ax.plot(x,y3,c='darkviolet',label='湿度',ls='-.',marker='^')
ax.tick_params(#top=True,bottom=True,left=True,right=True,
               width=1.5,direction='in',length=6,labelsize=15)
plt.legend(loc='upper left',fontsize=12)
ax2=ax.twinx()
ax2.plot(x,y1,c='orange',label='海拔',ls='--',marker='*')
ax.plot(x,y2,c='deepskyblue',label='温度',ls='-',marker='o')
ax.plot(x,y3,c='darkviolet',label='湿度',ls='-.',marker='^')
ax2.plot(x,y4,c='green',label='气压',ls=':',marker='*')
ax2.plot(x,y5,c='deeppink',label='海平面\n气压',ls='-.',marker='^')
ax2.set_xticks([0,1,2,3,4])
ax2.set_xticklabels(['兰州','白银','乌鞘岭','武威','张掖'])
ax2.tick_params(#top=True,bottom=True,left=True,right=True,
               width=1.5,direction='in',length=6,labelsize=15)
#ax2=ax.twinx()
plt.legend(loc='upper right',fontsize=12)
plt.show()
'''

'''
Python常用中文字体对应名称
黑体 SimHei
微软雅黑 Microsoft YaHei
微软正黑体 Microsoft JhengHei
新宋体 NSimSun
新细明体 PMingLiU
细明体 MingLiU
标楷体 DFKai-SB
仿宋 FangSong
楷体 KaiTi
仿宋_GB2312 FangSong_GB2312
楷体_GB2312 KaiTi_GB2312
'''


#经纬度分析
Z=xr.open_dataset('D:/GC/Z_1979-201908.nc')
z=Z.z.loc[:,500,30:45,90:110]
lon=z.longitude
lat=z.latitude

proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/本科课程学习/SX/甘肃省.shp'
GS=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='coral',facecolor='none')

ax.add_feature(GS,lw=1.5,zorder=2)
ax.set_extent([91,110,31,44])
#ax.add_feature(cfeature.LAND.with_scale('50m'))  #scale:110m\50m\10m
#ax.add_feature(cfeature.OCEAN.with_scale('50m'))
#grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.tick_params(size=20,labelsize=80)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linestyle='--')
gl.xlabels_top = True       #标签
gl.ylabels_right = True
gl.xlabels_bottom= True
gl.ylabels_left=True
gl.xlines = True           #网格线
gl.ylines=True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16, 'color':'black'}
gl.ylabel_style = {'size':16, 'color':'black'}
xx=[36.05,35.92,37.2,37.92,38.93]
yy=[103.66,104.30,102.85,102.63,100.46]
lon,lat=np.meshgrid(lon,lat)

vertices=[(103.66,36.05),(104.30,35.92),(102.85,37.2),(102.86,37.92),(100.46,38.93)]
codes=[Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]
path=mpath.Path(vertices,codes)


plt.rcParams['font.sans-serif']=['KaiTi'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

nameandstation={"兰州":[103.82,36.04]}
for key,value in nameandstation.items():
    ax.scatter(value[0] , value[1] , marker='*' , s=37 , color = "r" , zorder = 3)
    ax.text(value[0]+0.03 , value[1]-0.57 , key , fontsize = 15, color = "darkviolet")
nameandstation={"乌鞘岭":[102.85,37.2]}
for key,value in nameandstation.items():
    ax.scatter(value[0] , value[1] , marker='*' , s=37 , color = "r" , zorder = 3)
    ax.text(value[0]-2.7 , value[1]-0.37 , key , fontsize = 15, color = "darkviolet")
nameandstation={"白银":[104.18,36.55],"武威":[102.63,37.92],"张掖":[100.45,38.93]}
for key,value in nameandstation.items():
    ax.scatter(value[0] , value[1] , marker='*' , s=37 , color = "r" , zorder = 3)
    ax.text(value[0]+0.03 , value[1]+0.03 , key , fontsize = 15, color = "darkviolet")
plt.show()

