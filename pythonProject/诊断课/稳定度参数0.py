import numpy as np
import math
a=np.loadtxt('D:/ZD/DATA/239/500.txt')         # a 代表500hPa
b=np.loadtxt('D:/ZD/DATA/239/700.txt')         # b 代表700hPa
c=np.loadtxt('D:/ZD/DATA/239/850.txt')         # c 代表850hPa
g=9.8;pi=3.14;R=287;Cp=1004;L=2.5e6        #常数
###定义后面所需要的数组
T=np.empty((29,3));e=np.empty((29,3));TD=np.empty((29,3));H=np.empty((29,3))
q=np.empty((29,3));Tc=np.empty((29,3));theta_se=np.empty((29,3));theta=np.empty((29,3))
k=np.empty((29,1));r=np.empty((29,1));P=[500,700,850]
#开始循环29各站点，所有计算在一个循环里面
for i in np.arange(0,29):
    ah=a[i,3]*10 ; bh=b[i,3]*10 ; ch=c[i,3]*10
    H[i]=[ah,bh,ch]                            #位势高度
    at=a[i,4] ; bt=b[i,4] ; ct=c[i,4]
    T[i]=[at,bt,ct]                            #气温
    Tda=a[i,4]-a[i,5]/10 ; Tdb=b[i,4]-b[i,5]/10 ; Tdc=c[i,4]-c[i,5]/10
    TD[i]=[Tda,Tdb,Tdc]                        #露点温度
    ###  k指数  ###################################################################

    k[i]=c[i,4]-a[i,4]+Tdc-(b[i,4]-Tdb)

    at=at+273.15;bt=bt+273.15;ct=ct+273.15
    Tda=Tda+273.15;Tdb=Tdb+273.15;Tdc=Tdc+273.15
    a_theta=at*(1000/500)**0.286 ; b_theta=bt*(1000/700)**0.286 ; c_theta=ct*(1000/850)**0.286
    theta[i]=[a_theta,b_theta,c_theta]         #位温
    awd=a[i,6];bwd=b[i,6];cwd=c[i,6]           # 风向(a wind direction)
    aws=a[i,7];bws=b[i,7];cws=c[i,7]           # 风速(a wind speed)
    au=aws*math.cos(awd*pi/180)                # 高空u
    av=aws*math.sin(awd*pi/180)
    bu=bws*math.cos(bwd*pi/180)                # 中空u
    bv=bws*math.sin(bwd*pi/180)
    cu=cws*math.cos(cwd*pi/180)                # 低空u
    cv=cws*math.sin(cwd*pi/180)
    ###  理查孙数r  ################################################################

    r[i]=(g/b_theta)*((a_theta-c_theta)/(ah-ch))/(
        ((au-cu)/(ah-ch))**2+((av-cv)/(ah-ch))**2)

    #计算B
    Ba=0.622*L/Cp/Tda-1
    Bb=0.622*L/Cp/Tdb-1
    Bc=0.622*L/Cp/Tdc-1
    #计算Tc
    Tca=(Tda)*Ba/(Ba+math.log(at/Tda))
    Tcb=(Tdb)*Bb/(Bb+math.log(bt/Tdb))
    Tcc=(Tdc)*Bc/(Bc+math.log(ct/Tdc))
    Tc[i]=[Tca,Tcb,Tcc]
    at=at-273.15;bt=bt-273.15;ct=ct-273.15
    Tda=Tda-273.15;Tdb=Tdb-273.15;Tdc=Tdc-273.15
    for j in np.arange(0,3):
    #计算 q
        if TD[i,j]>-15:
            e[i,j]=6.1078*math.exp(17.26*TD[i,j]/(273.16+TD[i,j]-35.86))
        if -40<=TD[i,j]<-15:
            e0=6.1078*math.exp(17.26*TD[i,j]/(273.16+TD[i,j]-35.86))
            e1=6.1078*math.exp(21.8746*TD[i,j]/(273.16+TD[i,j]-7.66))
            e[i,j]=0.022*((80+2*TD[i,j])*e0-(30+2*TD[i,j])*e1)
        if TD[i,j]<-40:
            e[i,j]=6.1078*math.exp(21.8746*TD[i,j]/(273.16+TD[i,j]-7.66))
        q[i,j]=0.622*e[i,j]/P[j]
        theta_se[i,j]=theta[i,j]*math.exp(L*q[i,j]/Cp/Tc[i,j])  #假相当位温
###计算假相当位温差###################################################################
d_se=theta_se[:,0]-theta_se[:,2]
#将输出结果写入文件
import sys
sys.stdout=open('D:/ZD/稳定度参数0结果.log',mode='w',encoding='utf-8') #.log 或 .txt
print('k','\n',k)
print('r','\n',r)
print('d_tse','\n',d_se)
k=k.flatten()
r=r.flatten()
#将三个结果导入一个表格
from pandas import DataFrame
frame=DataFrame([k,r,d_se],index=['K指数','理查孙数','位温差'])
print(frame)