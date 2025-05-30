import numpy as np
import math
R=287;p5=50000;p8=85000;d=400000
Qx5=np.empty((11,10))
Qx8=np.empty((11,10))
Qy5=np.empty((11,10))
Qy8=np.empty((11,10))
H5=np.loadtxt('D:/ZD/DATA/555/H500.txt')           #500hPa数据
T5=np.loadtxt('D:/ZD/DATA/555/T500.txt')
U5=np.loadtxt('D:/ZD/DATA/555/U500.txt')
V5=np.loadtxt('D:/ZD/DATA/555/V500.txt')
H8=np.loadtxt('D:/ZD/DATA/555/H850.txt')           #850hPa数据
T8=np.loadtxt('D:/ZD/DATA/555/T850.txt')
U8=np.loadtxt('D:/ZD/DATA/555/U850.txt')
V8=np.loadtxt('D:/ZD/DATA/555/V850.txt')
#计算Q
#Qx=[Qx5,Qx8];p=[p5,p8]

for j in np.arange(1,9):
    Qx5[0,j]=-R/p5*((U5[0,j+1]-U5[0,j-1])*(T5[0,j+1]-T5[0,j-1])/(4*d**2)+
                    (V5[0,j+1]-V5[0,j-1])/(2*d)*(T5[1,j]-T5[0,j])/d)
    Qx5[10,j]=-R/p5*((U5[10,j+1]-U5[10,j-1])*(T5[10,j+1]-T5[10,j-1])/(4*d**2)+
                    (V5[10,j+1]-V5[10,j-1])/(2*d)*(T5[10,j]-T5[9,j])/d)
    Qy5[0,j]=-R/p5*((U5[1,j]-U5[0,j])/d*(T5[0,j+1]-T5[0,j-1])/(2*d)+
                    (V5[1,j]-V5[0,j])*(T5[1,j]-T5[0,j])/d**2)
    Qy5[10,j]=-R/p5*((U5[10,j]-U5[9,j])/d*(T5[10,j+1]-T5[10,j-1])/(2*d)+
                     (V5[10,j]-V5[9,j])*(T5[10,j]-T5[9,j])/d**2)

    Qx8[0,j]=-R/p8*((U8[0,j+1]-U8[0,j-1])*(T8[0,j+1]-T8[0,j-1])/(4*d**2)+
                    (V8[0,j+1]-V8[0,j-1])/(2*d)*(T8[1,j]-T8[0,j])/d)
    Qx8[10,j]=-R/p8*((U8[10,j+1]-U8[10,j-1])*(T8[10,j+1]-T8[10,j-1])/(4*d**2)+
                    (V8[10,j+1]-V8[10,j-1])/(2*d)*(T8[10,j]-T8[9,j])/d)
    Qy8[0,j]=-R/p8*((U8[1,j]-U8[0,j])/d*(T8[0,j+1]-T8[0,j-1])/(2*d)+
                    (V8[1,j]-V8[0,j])*(T8[1,j]-T8[0,j])/d**2)
    Qy8[10,j]=-R/p8*((U8[10,j]-U8[9,j])/d*(T8[10,j+1]-T8[10,j-1])/(2*d)+
                     (V8[10,j]-V8[9,j])*(T8[10,j]-T8[9,j])/d**2)
for i in np.arange(1,10):
    Qx5[i,0]=-R/p5*((U5[i,1]-U5[i,0])/d*(T5[i,1]-T5[i,0])/d+
                    (V5[i,1]-V5[i,0])/d*(T5[i+1,0]-T5[i-1,0])/(2*d))
    Qx5[i,9]=-R/p5*((U5[i,9]-U5[i,8])/d*(T5[i,9]-T5[i,8])/d+
                    (V5[i,9]-V5[i,8])/d*(T5[i+1,9]-T5[i-1,9])/(2*d))
    Qy5[i,0]=-R/p5*((U5[i+1,0]-U5[i-1,0])/(2*d)*(T5[i,1]-T5[i,0])/d+
                    (V5[i+1,0]-V5[i-1,0])/(2*d)*(T5[i+1,0]-T5[i-1,0])/(2*d))
    Qy5[i,9]=-R/p5*((U5[i+1,9]-U5[i-1,9])/(2*d)*(T5[i,9]-T5[i,8])/d+
                    (V5[i+1,9]-V5[i-1,9])/(2*d)*(T5[i+1,9]-T5[i-1,9])/(2*d))

    Qx8[i,0]=-R/p8*((U8[i,1]-U8[i,0])/d*(T8[i,1]-T8[i,0])/d+
                    (V8[i,1]-V8[i,0])/d*(T8[i+1,0]-T8[i-1,0])/(2*d))
    Qx8[i,9]=-R/p8*((U8[i,9]-U8[i,8])/d*(T8[i,9]-T8[i,8])/d+
                    (V8[i,9]-V8[i,8])/d*(T8[i+1,9]-T8[i-1,9])/(2*d))
    Qy8[i,0]=-R/p8*((U8[i+1,0]-U8[i-1,0])/(2*d)*(T8[i,1]-T8[i,0])/d+
                    (V8[i+1,0]-V8[i-1,0])/(2*d)*(T8[i+1,0]-T8[i-1,0])/(2*d))
    Qy8[i,9]=-R/p8*((U8[i+1,9]-U8[i-1,9])/(2*d)*(T8[i,9]-T8[i,8])/d+
                    (V8[i+1,9]-V8[i-1,9])/(2*d)*(T8[i+1,9]-T8[i-1,9])/(2*d))
for i in np.arange(1,10):
    for j in np.arange(1,9):
        Qx5[i,j]=-R/p5*((U5[i,j+1]-U5[i,j-1])*(T5[i,j+1]-U5[i,j-1])/(4*d**2)+
                        (V5[i,j+1]-V5[i,j-1])/(2*d)*(T5[i+1,j]-T5[i-1,j])/(2*d))
        Qx8[i,j]=-R/p8*((U8[i,j+1]-U8[i,j-1])*(T8[i,j+1]-U8[i,j-1])/(4*d**2)+
                        (V8[i,j+1]-V8[i,j-1])/(2*d)*(T8[i+1,j]-T8[i-1,j])/(2*d))
        Qy5[i,j]=-R/p5*((U5[i+1,j]-U5[i-1,j])*(T5[i,j+1]-T5[i,j-1])/(4*d**2)+
                        (V5[i+1,j]-V5[i-1,j])*(T5[i+1,j]-T5[i-1,j])/(4*d**2))
        Qy8[i,j]=-R/p8*((U8[i+1,j]-U8[i-1,j])*(T8[i,j+1]-T8[i,j-1])/(4*d**2)+
                        (V8[i+1,j]-V8[i-1,j])*(T8[i+1,j]-T8[i-1,j])/(4*d**2))
Qx5[0,0]=(Qx5[0,1]+Qx5[1,0])/2
Qx5[0,9]=(Qx5[0,8]+Qx5[1,9])/2
Qx5[10,0]=(Qx5[9,0]+Qx5[10,1])/2
Qx5[10,9]=(Qx5[10,8]+Qx5[9,9])/2
Qx8[0,0]=(Qx8[0,1]+Qx8[1,0])/2
Qx8[0,9]=(Qx8[0,8]+Qx8[1,9])/2
Qx8[10,0]=(Qx8[9,0]+Qx8[10,1])/2
Qx8[10,9]=(Qx8[10,8]+Qx8[9,9])/2
Qy5[0,0]=(Qy5[0,1]+Qy5[1,0])/2
Qy5[0,9]=(Qy5[0,8]+Qy5[1,9])/2
Qy5[10,0]=(Qy5[9,0]+Qy5[10,1])/2
Qy5[10,9]=(Qy5[10,8]+Qy5[9,9])/2
Qy8[0,0]=(Qy8[0,1]+Qy8[1,0])/2
Qy8[0,9]=(Qy8[0,8]+Qy8[1,9])/2
Qy8[10,0]=(Qy8[9,0]+Qy8[10,1])/2
Qy8[10,9]=(Qy8[10,8]+Qy8[9,9])/2
print('Qx5','\n',Qx5)
print('Qx8','\n',Qx8)
print('Qy5','\n',Qy5)
print('Qy8','\n',Qy8)
f=np.empty((11,10))
theta5=np.empty((11,10))
theta8=np.empty((11,10))
sigma5=np.empty((11,10))
sigma8=np.empty((11,10))
a=6370;pi=3.14
d_lat=(d/a)*pi/180
T5=T5+273.15
T8=T8+273.15
for i in np.arange(0,11):
  for j in np.arange(0,10):
    lat=60-i*d_lat
    f[i,j]=2*(2*pi/86400)*math.sin(lat)
    r=-(T5[i,j]-T8[i,j])/(H5[i,j]-H8[i,j])
    theta5[i,j]=T5[i,j]*(1000/500)**0.286
    theta8[i,j]=T8[i,j]*(1000/850)**0.286
    sigma5[i,j]=-(R*T5[i,j]/500/theta5[i,j])*((theta5[i,j]-theta8[i,j])/(500-850))
    sigma8[i,j]=-(R*T8[i,j]/850/theta8[i,j])*((theta5[i,j]-theta8[i,j])/(500-850))
sigma=(sigma5+sigma8)/2
T5=T5-273.15
T8=T8-273.15
F5=np.empty((11,10))
F8=np.empty((11,10))
for j in np.arange(1,9):
    F5[0,j]=-2*((Qx5[0,j+1]-Qx5[0,j-1])/(2*d)+(Qy5[1,j]-Qy5[0,j])/d)/sigma5[0,j]
    F5[10,j]=-2*((Qx5[10,j+1]-Qx5[10,j-1])/(2*d)+(Qy5[10,j]-Qy5[9,j])/d)/sigma5[10,j]
    F8[0,j]=-2*((Qx8[0,j+1]-Qx8[0,j-1])/(2*d)+(Qy8[1,j]-Qy8[0,j])/d)/sigma8[0,j]
    F8[10,j]=-2*((Qx8[10,j+1]-Qx8[10,j-1])/(2*d)+(Qy8[10,j]-Qy8[9,j])/d)/sigma8[10,j]

for i in np.arange(1,10):
    F5[i,0]=-2*((Qx5[i,1]-Qx5[i,0])/d+(Qy5[i+1,0]-Qy5[i-1,0])/(2*d))/sigma5[i,0]
    F5[i,9]=-2*((Qx5[i,9]-Qx5[i,8])/d+(Qy5[i+1,9]-Qy5[i-1,9])/(2*d))/sigma5[i,9]
    F8[i,0]=-2*((Qx8[i,1]-Qx8[i,0])/d+(Qy8[i+1,0]-Qy8[i-1,0])/(2*d))/sigma8[i,0]
    F8[i,9]=-2*((Qx8[i,9]-Qx8[i,8])/d+(Qy8[i+1,9]-Qy8[i-1,9])/(2*d))/sigma8[i,9]
for i in np.arange(1,10):
    for j in np.arange(1,9):
        F5[i,j]=-2*((Qx5[i,j+1]-Qx5[i,j-1])/(2*d)+(Qy5[i+1,j]-Qy5[i-1,j])/(2*d))/sigma5[i,j]
        F8[i,j]=-2*((Qx8[i,j+1]-Qx8[i,j-1])/(2*d)+(Qy8[i+1,j]-Qy8[i-1,j])/(2*d))/sigma8[i,j]
F5[0,0]=(F5[0,1]+F5[1,0])/2
F5[0,9]=(F5[0,8]+F5[1,9])/2
F5[10,0]=(F5[9,0]+F5[10,1])/2
F5[10,9]=(F5[9,9]+F5[10,8])/2
F8[0,0]=(F8[0,1]+F8[1,0])/2
F8[0,9]=(F8[0,8]+F8[1,9])/2
F8[10,0]=(F8[9,0]+F8[10,1])/2
F8[10,9]=(F8[9,9]+F8[10,8])/2
F_5=d**2*F5
F_8=d**2*F8
n=10   #将n暂定为迭代次数
omega5=np.zeros((11,10))
omega8=np.zeros((11,10))
lmd5=2*f*f/(p5*(p5-1000)*sigma5)
lmd8=2*f*f/(p8*(p8-1000)*sigma8)
R=np.empty((11,10))

for i in np.arange(1,10):
    for j in np.arange(1,9):
      r=0
      while abs(omega5[i,j])<10**(-2):
         R[i,j]=omega5[i+1,j]+omega5[i-1,j]+omega5[i,j+1]+omega5[i,j-1]-(4-lmd5[i,j]*d*d)*omega5[i,j]-F_5[i,j]
         omega5[i,j]=omega5[i,j]+R[i,j]/(4-lmd5[i,j]*d*d)
         omega5[0,:]=omega5[1,:]
         omega5[10,:]=omega5[9,:]
         omega5[:,0]=omega5[:,1]
         omega5[:,9]=omega5[:,8]
         omega5[0,0]=(omega5[1,0]+omega5[0,1])/2
         omega5[0,9]=(omega5[1,9]+omega5[0,8])/2
         omega5[10,0]=(omega5[9,0]+omega5[10,1])/2
         omega5[10,9]=(omega5[10,8]+omega5[9,9])/2
         r=r+1
         if r>1000:
             break
      while abs(omega8[i,j])<10**(-2):
         R[i,j]=omega8[i+1,j]+omega8[i-1,j]+omega8[i,j+1]+omega8[i,j-1]-(4-lmd8[i,j]*d*d)*omega8[i,j]-F_8[i,j]
         omega8[i,j]=omega8[i,j]+R[i,j]/(4-lmd8[i,j]*d*d)
         omega8[0,:]=omega8[1,:]
         omega8[10,:]=omega8[9,:]
         omega8[:,0]=omega8[:,1]
         omega8[:,9]=omega8[:,8]
         omega8[0,0]=(omega8[1,0]+omega8[0,1])/2
         omega8[0,9]=(omega8[1,9]+omega8[0,8])/2
         omega8[10,0]=(omega8[9,0]+omega8[10,1])/2
         omega8[10,9]=(omega8[10,8]+omega8[9,9])/2
         r=r+1
         if r>1000:
             break
#print(r)
print('omega5','\n',omega5)
print('omega8','\n',omega8)
import numpy as np
import matplotlib.pyplot as plt
clevs = np.arange(10e-8,10e-6,10e-8)  #set by max and min of data
x=np.arange(1,11)
y=np.arange(1,12)
X,Y=np.meshgrid(x,y)
fig,ax=plt.subplots(dpi=100)
plt.contourf(X, Y, omega5,clevs,cmap=plt.cm.coolwarm,extend='both')
#plt.title('MERRA-2 2m Wind Speed and Direction, 00Z 1 June 2010', size=16)
cb = plt.colorbar(ax=ax, orientation="vertical", extend='both',pad=0.04, aspect=20, shrink=0.8)#,drawedges=True
#horizontal横vertical竖，shrink收缩比例，ax、cax位置，aspect长宽比，pad距子图，extend两端扩充，extendfrac扩充长度，extendrect扩充形状True，spacing，
cb.set_label(' ',size=11,rotation=0,labelpad=10)
cb.ax.tick_params(labelsize=7)  #参数刻度样式
# Overlay wind vectors
plt.quiver(X, Y, Qx5, Qy5, width=0.006,scale=100000, color='k')
#width决定箭头箭轴宽度，scale决定箭杆长度  scale=420
#plt.savefig('F:/Rpython/lp28/plot29.1.png',dpi=1200)
plt.show()