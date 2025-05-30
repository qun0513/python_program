import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
ds = xr.open_dataset('D:/PY/t2m_daily_2000_2009.nc')
#数据处理
t2m=ds['t2m']-273.15
t2m_time=t2m.time
lon=t2m.longitude
lat=t2m.latitude
#计算该区域内多年平均的热浪发生频率
temp_data=[]
for i in lat:
    for j in lon:
        data=t2m.sel(longitude=j,latitude=i)
        temp_data.append(np.array(data))
heat_wave=[]
for temp in temp_data:
    day=0
    sum_day=0
    for i in range(3653):
        if temp[i]>20:
            day+=1
        else:
            if day>=5:
                sum_day+=day
            day=0
    heat_wave.append(sum_day/3653)
print(heat_wave)
heat_wave=np.array(heat_wave).reshape(41,41)
print(heat_wave)
#绘制中国地图
from cartopy.io.shapereader import Reader
proj=ccrs.PlateCarree()
ax=plt.axes(projection=proj)
shape_path='D:/PY/china-myclass.shp'
china=cfeature.ShapelyFeature(Reader(shape_path).geometries(),proj,edgecolor='k',facecolor='none')
ax.add_feature(china,lw=0.5,zorder=2)
grid_lines=ax.gridlines(crs=proj,draw_labels=True,color='g',linestyle='--',alpha=0.5)
ax.add_feature(cfeature.LAND.with_scale('50m'))  #scale:110m\50m\10m
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
#绘制空间分布图像
clevs=np.linspace(-0.5,0.5,10)
contourf = ax.contourf(lon,lat,heat_wave,cmap=plt.cm.coolwarm)
m=ax.contourf(lon,lat,heat_wave,cmap=plt.cm.jet)
plt.colorbar(m,ax=ax,pad=0.15,
             ticks=[0,0.1,0.2,0.3,0.4],
             label='色条',
             )  #第一个参数：使用哪个颜色映射表；
                                       # 第二个参数：放置在哪个子图
                                       #cax参数可以在多子图中妙用
#ax.set_yticks([])
#ax.set_ytickslabels([])

#ax2=cb.ax#召唤出cb的ax属性并简称为ax2,这时ax2即视为一个子图
'''
    ###############################以下为添加县市名称和点########################################
nameandstation={"恩施":[109.5,30.2],"利川":[109,30.3],"巴东":[110.34,31.04],"建始":[109.72,30.6],"宣恩":[109.49,29.987],"来凤":[109.407,29.493],"咸丰":[109.14,29.665],"鹤峰":[110.034,29.89]}
for key,value in nameandstation.items():
    ax.scatter(value[0] , value[1] , marker='.' , s=90 , color = "k" , zorder = 3)
    ax.text(value[0]-0.07 , value[1]+0.03 , key , fontsize = 12 , color = "k")
'''

#ax.set_extent([-30,150,0,90])
ax.set_title('Spatial distribution of annual average heat wave frequency')

plt.show()

'''
with open('D:/ZD/dddff.txt',encoding='utf-8') as file_2:
    df=file_2.read()
    print(df.rstrip())
'''