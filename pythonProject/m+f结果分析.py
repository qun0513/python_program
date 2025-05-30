import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
m=np.loadtxt('D:/FORTRAN/M.txt')
f=np.loadtxt('D:/FORTRAN/F.txt')
print(m)
print(f)
n=np.empty((16,20))
for i in np.arange(0,16):
    for j in np.arange(0,20):
        n[i,j]=m[15-i,j]
x=np.arange(1,21)
y=np.arange(1,17)
X,Y=np.meshgrid(x,y)
fig=plt.figure(figsize=(55,40),dpi=50)
ax=plt.axes()
levels=np.linspace(0.965816,1.117631,100)# 0.965816,1.117631,100 0.000052,0.000139,100
plt.contourf(X,Y,n,levels,cmap=mpl.cm.jet,extend='both')
ax.set_xlabel('x',size=80)
ax.set_ylabel('y',size=80)
ax.set_title('地图放大系数m',fontsize=100,pad=40)  #地图放大系数m  科氏参数f
ax.tick_params(size=20,labelsize=80)

cb = plt.colorbar(ax=ax,orientation="vertical", pad=0.05, aspect=16, shrink=1)
cb.set_label('m',size=14,rotation=0,labelpad=40,fontsize=80)
cb.ax.tick_params(labelsize=80)

plt.rcParams['font.sans-serif']=['FangSong'] #用来正常显示中文标签 SimHei
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#plt.savefig('D:/FORTRAN/地图放大系数m.png')
plt.show()