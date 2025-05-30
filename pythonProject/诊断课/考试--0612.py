import numpy as np
import math
import xarray as xr
#将输出结果写入文件
import sys
sys.stdout=open('D:/ZD/位涡结果.txt',mode='w',encoding='utf-8') #.log 或 .txt
g=9.8;pi=3.14;a=6370000
f=np.empty((41,21))
dx=np.empty(41)          #x方向网格距
theta1=np.empty((41,21)) #位温175hPa
theta2=np.empty((41,21)) #位温225hPa
vort=np.empty((41,21))   #vorticity 涡度
PV=np.empty((41,21))     #位涡
'''   UV=xr.open_dataset('D:/ZD/diagonal/uv_200hpa_era5_202001.nc')
T=xr.open_dataset('D:/ZD/diagonal/T_200hpa_era5_202001.nc')   '''
#读取数据
u2=np.loadtxt('D:/ZD/diagonal/u200.txt')
v2=np.loadtxt('D:/ZD/diagonal/v200.txt')
t1=np.loadtxt('D:/ZD/diagonal/t175.txt')
t2=np.loadtxt('D:/ZD/diagonal/t225.txt')
lat=np.loadtxt('D:/ZD/diagonal/lat.txt')
lon=np.loadtxt('D:/ZD/diagonal/lon.txt')
#计算过程
dy=2*pi*a*(0.25/360)                                   #y方向网格距
for i in np.arange(0,41):
  dx[i]=2*pi*(a*math.cos((lat[i])*pi/180))*(0.25/360)  #x方向网格距
def DX(data):    #定义函数 对x方向偏导
    Dx=np.empty((41,21))
    for i in np.arange(0,41):
        for j in np.arange(0,21):
            if j==0:
                Dx[i,0]=(data[i,1]-data[i,0])/dx[i]
            elif j==20:
                Dx[i,20]=(data[i,20]-data[i,19])/dx[i]
            else:
                Dx[i,j]=(data[i,j+1]-data[i,j-1])/(2*dx[i])
    return Dx
def DY(data):    #定义函数 对y方向偏导
    Dy=np.empty((41,21))
    for i in np.arange(0,41):
        for j in np.arange(0,21):
            if i==0:
                Dy[0,j]=(data[0,j]-data[1,j])/dy
            elif i==40:
                Dy[40,j]=(data[39,j]-data[40,j])/dy
            else:
                Dy[i,j]=(data[i-1,j]-data[i+1,j])/(2*dy)
    return Dy
vort_x=DX(v2)
vort_y=DY(u2)
for i in np.arange(0,41):
  for j in np.arange(0,21):
    f[i,j]=2*(2*pi/86400)*math.sin(lat[i]*pi/180)     #地转涡度
    theta1[i,j]=t1[i,j]*(100000/17500)**0.286         #位温175hPa
    theta2[i,j]=t2[i,j]*(100000/22500)**0.286         #位温225hPa
    vort[i,j]=vort_x[i,j]-vort_y[i,j]                 #相对涡度
    PV[i,j]=-g*(vort[i,j]+f[i,j])*((theta1[i,j]-theta2[i,j])/(17500-22500))  #位涡
print('位势涡度PV','\n',PV)
