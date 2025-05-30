import math
import numpy as np
df=np.loadtxt('D:/ZD/dddff.txt')
H=np.loadtxt('D:/ZD/HHH.txt')
k=0.7156;a=6370;p=3.14159;g=9.8
l0=(a*math.sin(p/3))/k
u1=[]
v1=[]
ccc=[]
mm=[]
for i in np.arange(0,5):
    ly=l0-(4-i)*100
    for j in np.arange(0,6,1):
        wd=df[i,2*j]    #风向（wind direction）
        ws=df[i,2*j+1]  #风速（wind speed）
        lx=j*100
        l=(lx**2+ly**2)**(1/2)
        #求m
        t=(l/l0)**(1/k)*math.tan(p/6)  # t 为 tan（）
        s=(2*t)/(1+t**2)  # s 为 sin（*）
        m=(k*l)/(a*s)
        mm.append(m)
        c=(1-s**2)**1/2
        ccc.append(c)  #纬度的正弦/余纬的余弦
        #求订正后的风速
        ss=ly/l  # s 为 sin(**)
        cc=lx/l  # c 为 cos()
        u=ws*math.sin(p*wd/180-p)
        v=ws*math.cos(p*wd/180-p)
        u0=u*ss-v*cc  #订正后的 u
        u1.append(u0)
        v0=v*ss+u*cc  #订正后的 v
        v1.append(v0)
u2=np.array(u1).reshape(5,6)
v2=np.array(v1).reshape(5,6)
m=np.array(mm).reshape(5,6)
ccc=np.array(ccc).reshape(5,6)
#求解
RR=[]  #实测风涡度
TT=[]  #地转风涡度
D=[]  #散度
for i in [1,2,3]:
    for j in [1,2,3,4]:
        f=(2*2*p/86400)*ccc[i,j]
        rr=(m[i,j]/(2*100000))*((v2[i,j+1]-v2[i,j-1])-(u2[i-1,j]-u2[i+1,j]))
        tt=((g*m[i,j])/(f*100000*100000))*(H[i+1,j]+H[i-1,j]+H[i,j+1]+H[i,j-1]-4*H[i,j])
        d=(m[i,j]/(2*100000))*(u2[i,j+1]-u2[i,j-1]+v2[i-1,j]-v2[i+1,j])
        RR.append(rr)
        TT.append(tt)
        D.append(d)
RR=np.array(RR).reshape(3,4)
TT=np.array(TT).reshape(3,4)
D=np.array(D).reshape(3,4)
print(RR,'\n')
print(TT,'\n')
print(D,'\n')
