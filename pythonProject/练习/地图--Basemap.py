import warnings
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
'''
map=Basemap()
map.drawcoastlines()  #海岸线
map.drawcountries()  #国界线
map.fillcontinents(color='coral',lake_color='aqua')  #为大陆内部着色
#map.drawrivers()  #绘制河流
#map.drawstates()  #绘制北美的州界
#map.drawlsmask()  #绘制高分辨率海陆掩码作为图像，指定土地和海洋颜色
#map.bluemarble()  #绘制一张NASA蓝色大理石图像作为地图背景。
#map.shadedrelief()  #绘制一个阴影浮雕图像作为地图背景。
map.etopo()  #绘制etopo浮雕图像作为地图背景
#map.warpimage()  #使用abitrary图像作为地图背景。图像必须是全球的，从国际数据线向东，南极向北，以纬度/经度坐标覆盖世界。
'''

map=Basemap(llcrnrlon=70,llcrnrlat=3,urcrnrlon=140,urcrnrlat=54,
            resolution='i',projection='cass',lat_0=25,lon_0=105)

#Basemap的各类参数
#resolution为分辨率：c,l,i,h,f,none
'''
Keyword	Description
llcrnrlon	longitude of lower left hand corner of the desired map domain (degrees).
llcrnrlat	latitude of lower left hand corner of the desired map domain (degrees).
urcrnrlon	longitude of upper right hand corner of the desired map domain (degrees).
urcrnrlat	latitude of upper right hand corner of the desired map domain (degrees).
or these

Keyword	Description
width	width of desired map domain in projection coordinates (meters).
height	height of desired map domain in projection coordinates (meters).
lon_0	center of desired map domain (in degrees).
lat_0	center of desired map domain (in degrees).
'''

map.etopo()
map.drawcoastlines()
plt.show()