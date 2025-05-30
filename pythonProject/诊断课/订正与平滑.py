import numpy as np
import math
#读取文件
df=np.loadtxt('D:/ZD/dddff.txt')
H=np.loadtxt('D:/ZD/HHH.txt')
T=np.loadtxt('D:/ZD/ttt.txt')
#常数
k=0.7156;a=6370;p=3.14
l0=(a*math.sin(p/3))/k
#风速订正
u1=[]
v1=[]
for i in np.arange(0,5):
    ly=l0-i*100
    for j in np.arange(0,6,1):
        wd=df[i,2*j]    #风向（wind direction）
        ws=df[i,2*j+1]  #风速（wind speed）
        lx=j*100
        l=(lx**2+ly**2)**(1/2)
        s=ly/l  # s 为 sin()
        c=lx/l  # c 为 cos()
        u=ws*math.sin(p*wd/180-p)
        v=ws*math.cos(p*wd/180-p)
        u0=u*s-v*c  #订正后的 u
        u1.append(u0)
        v0=v*s+u*c  #订正后的 v
        v1.append(v0)
        print(u0,v0)
u2=np.array(u1).reshape(5,6)
v2=np.array(v1).reshape(5,6)
#九点平滑
s=1/2
#定义函数
def LS(H):   #LS (level and smooth)
    r=[]
    for i in [1,2,3]:
        for j in [1,2,3,4]:
            m=H[i+1,j]+H[i-1,j]+H[i,j-1]+H[i,j+1]-4*H[i,j]
            n=H[i+1,j+1]+H[i+1,j-1]+H[i-1,j+1]+H[i-1,j-1]-4*H[i,j]
            o=H[i,j]+(s*(1-s)/2)*m+(s**2)/4*n
            r.append(o)
    return np.array(r).reshape(3,4)
print(LS(H))
print(LS(u2))
print(LS(v2))
print(LS(T))

'''
import numpy as np
from pandas import DataFrame
import math
#读取文件
df=np.loadtxt('D:/ZD/dddff.txt')
H=np.loadtxt('D:/ZD/HHH.txt')
T=np.loadtxt('D:/ZD/ttt.txt')
#常数
k=0.7156;r=6370;p=3.14
l0=(r*math.sin(p/3))/k
#订正与平滑
u1=[]
v1=[]
for i in np.arange(0,5):
    ly=l0-i*100
    for j in np.arange(0,6,1):
        wd=df[i,2*j]
        ws=df[i,2*j+1]
        lx=j*100
        l=(lx**2+ly**2)**(1/2)
        s=ly/l  #s 为 sin()
        c=lx/l  #c 为 cos()
        u=ws*math.sin(p*wd/180-p)
        v=ws*math.cos(p*wd/180-p)
        u0=u*s-v*c  #订正后的 u
        u1.append(u0)
        v0=v*s+u*c  #订正后的 v
        v1.append(v0)
        print(u0,v0)
u2=np.array(u1).reshape(5,6)
v2=np.array(v1).reshape(5,6)
#九点平滑
s=1/2
Hij=[]
Uij=[]
Vij=[]
Tij=[]
for i in [1,2,3]:
    for j in [1,2,3,4]:
        #位势高度九点平滑
        p=H[i+1,j]+H[i-1,j]+H[i,j-1]+H[i,j+1]-4*H[i,j]
        q=H[i+1,j+1]+H[i+1,j-1]+H[i-1,j+1]+H[i-1,j-1]-4*H[i,j]
        hij=H[i,j]+(s*(1-s)/2)*p+(s**2)/4*q
        Hij.append(hij)
        #u0九点平滑
        m=u2[i+1,j]+u2[i-1,j]+u2[i,j-1]+u2[i,j+1]-4*u2[i,j]
        n=u2[i+1,j+1]+u2[i+1,j-1]+u2[i-1,j+1]+u2[i-1,j-1]-4*u2[i,j]
        uij=u2[i,j]+(s*(1-s)/2)*m+(s**2)/4*n
        Uij.append(uij)
        #v0九点平滑
        x=v2[i+1,j]+v2[i-1,j]+v2[i,j-1]+v2[i,j+1]-4*v2[i,j]
        y=v2[i+1,j+1]+v2[i+1,j-1]+v2[i-1,j+1]+v2[i-1,j-1]-4*v2[i,j]
        vij=v2[i,j]+(s*(1-s)/2)*x+(s**2)/4*y
        Vij.append(vij)
        #t九点平滑
        k=T[i+1,j]+T[i-1,j]+T[i,j-1]+T[i,j+1]-4*T[i,j]
        l=T[i+1,j+1]+T[i+1,j-1]+T[i-1,j+1]+T[i-1,j-1]-4*T[i,j]
        tij=T[i,j]+(s*(1-s)/2)*k+(s**2)/4*l
        Tij.append(tij)
Hij=np.array(Hij).reshape(3,4)
Uij=np.array(Uij).reshape(3,4)
Vij=np.array(Vij).reshape(3,4)
Tij=np.array(Tij).reshape(3,4)
print(Hij)
print(Uij)
print(Vij)
print(Tij)
'''