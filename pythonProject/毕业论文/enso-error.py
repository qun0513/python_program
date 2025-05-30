import numpy as np
from pandas import DataFrame
import pandas as pd

#import datetime
#from datetime import timedelta
#import string
b=pd.read_table('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs1.txt',
                sep=r'\s+',header=6)
#print(b)
#---sep参数，很重要---  \t，为逗号  r'\s+',为多个空格
c=b.iloc[5818::,:]
#print('c','\n',c)

c['date'] =c['Date/Time'].str.split('/').str.get(0)
c['time'] =c['Date/Time'].str.split('/').str.get(1)
#print(c.date)
c['date']=pd.to_datetime(c['date'],format='%d-%m-%Y')

y2017=c[c['date'].dt.year.isin([2017])]
y2018=c[c['date'].dt.year.isin([2018])]

z=['012hT01','024hT01','036hT01','048hT01','072hT01','096hT01','120hT01']
z1=['012hI01','024hI01','036hI01','048hI01','072hI01','096hI01','120hI01']
error=[];number=[];intensity=[]
sd=[];r=[];f=[]
for i in np.arange(0,19):
    y=c[c['date'].dt.year.isin([2001+i])]   # 01-16年的error

    m=0
    yind=y['STMID'].value_counts()
    yin=list(yind.index)
    length=len(yin)

    # 强度
    for j in np.arange(0,7):
        ynew = list(y[z1[j]])
        ylen = len(ynew)
        yintensities = 0;ynum0 = 0

        int=[]
        for jj in np.arange(0, ylen):
            if ynew[jj] != -9999.0:
                yintensities = yintensities + ynew[jj]
                ynum0 = ynum0 + 1               # 每一年、每一个提前期的  个数
                int.append(ynew[jj])           # 每一年、每一个提前期的  每个error
        if ynum0==0:
            intensity=0
        else:
            yintensity = yintensities / ynum0      # 每一年、每一个提前期的  error平均

        intensity.append(yintensity)

    # error
    for k in np.arange(0,7):
        ynew=list(y[z[k]])
        ylen=len(ynew)
        yerrors=0;ynum=0

        err = []
        for kk in np.arange(0,ylen):
            if ynew[kk]!=-9999.0:
                yerrors=yerrors+ynew[kk]
                ynum=ynum+1            #每一年、每一个提前期的  个数
                err.append(ynew[kk])   #每一年、每一个提前期的  每个error
        yerror=yerrors/ynum            #每一年、每一个提前期的  error平均

        error.append(yerror)
        number.append(ynum)

        sdi=np.std(err,ddof=1)     #样本标准差
        sd.append(sdi)

        rx=0;ry=0
        for rr in np.arange(0,len(err)-1):
            rx=rx+(err[rr]-yerror)*(err[rr+1]-yerror)
            ry=ry+(err[rr]-yerror)**2
        ry=ry+(err[len(err)-1]-yerror)**2
        ri=rx/ry
        r.append(ri)               #自相关系数

        fi=np.sqrt((1+ri)/(1-ri))  #f
        f.append(fi)
intensity=np.array(intensity).reshape(19,7)  #------------------------------
number1=np.array(number).reshape(19,7)
error1=np.array(error).reshape(19,7)
sd=np.array(sd).reshape(19,7)
r=np.array(r).reshape(19,7)
f=np.array(f).reshape(19,7)
#print('error1','\n',error1)
#print('number1','\n',number1)
#print('intensity','\n',intensity)

sem1=np.empty((19,7))
num_ind=np.empty((19,7))
w=np.empty((19,5))
for i in np.arange(0,19):
    for j in np.arange(0,7):
        sem1[i,j]=sd[i,j]*f[i,j]/np.sqrt(number1[i,j])  # sem
        num_ind[i,j]=number1[i,j]/(f[i,j])**2     # independent cases
    for k in np.arange(2,7):
        w[i,k-2]=sem1[i,k]/sum(sem1[i,2::])                 # w




'''
with open('D:/毕业设计/error1.txt','w') as outfile:
    np.savetxt(outfile,error1)
with open('D:/毕业设计/sem1.txt','w') as outfile:
    np.savetxt(outfile,sem1)
'''


days=np.array([2.625,2.83333333,4.33333333,3.54166667,3.41666667,3.5,
 4.33333333, 4.66666667,3.79166667, 4.5, 4.08333333,5.25,4.83333333,
 4.29166667, 4.16666667,5.125,5.7832,4.7716])  #---------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

oni=pd.read_table('D:/毕业设计/尼诺指数.txt',sep=r'\s+',header=0)
#b=pd.read_table('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt',
#                sep=r'\s+',header=7)

on_new=np.loadtxt('D:/毕业设计/尼诺指数1.txt')
print(on_new)
on7=on_new[6::12,2]
on8=on_new[7::12,2]
on9=on_new[8::12,2]
on10=on_new[9::12,2]

#print('on7',on7)
#print('on7',on7[17])
#print(np.mean(on7+on8+on9+on10,axis=1))
ninuo=np.empty((19))
for i in np.arange(0,19):
    ninuo[i]=(on7[i]+on8[i]+on9[i]+on10[i])/4
print('ninuo','\n',ninuo)

y=oni['Nino34'].groupby(oni['Year'])
nindex=y.mean()
parameter51=np.array([[62.04347,0.73623,22.68116],
                      [56.38296,0.76915,22.75318],
                      [61.64586,0.52153,17.21529],
                      [47.65495,0.69649,21.05847],
                      [47.06239,0.75312,19.62496],#5
                      [41.86888,0.79344,18.94755],
                      [42.96438,0.66429,16.32146],
                      [44.13354,0.60890,15.04451],
                      [33.14947,0.85747,16.45979],
                      [45.17931,0.61195,13.05977],#10
                      [30.88777,0.84439,14.35511],
                      [32.18740,0.70937,12.27496],
                      [48.96165,0.59233,7.84659],
                      [27.10280,0.87352,10.36760],
                      [31.27481,0.81832,9.09160],#15
                      [28.84309,0.72287,7.61436],
                      [25.00396,0.72039,8.20158],
                      [27.43023,0.78702,6.77209],
                      [25.20376,0.73539,8.10238]])
#print(np.array(nindex)[19:36])
#print(nindex)
#print(nindex[19:36])
on=np.array(nindex)[19:38]  #------------------------------------------
alpha=parameter51[:,1]*2    #------------------------------------------
#print('尼诺指数',on)
print('24h误差增长率',alpha)

import scipy
from scipy import signal
days=scipy.signal.detrend(days)    #去趋势

#归一化
from sklearn.preprocessing import scale
on_scale=scale(on)                      # ENSO  尼诺指数

alpha_scale=scale(alpha)                # 误差增长率
days_scale=scale(days)                  # 可预测时间
intensity_scale=scale(intensity[:,3])   # 强度
number_scale=scale(number1[:,0])
#print(on_scale)
#print(alpha_scale)

#相关性
on1=pd.Series(on)
onn=pd.Series(ninuo)
alpha1=pd.Series(alpha)            #1
days1=pd.Series(days)              #2
intensity1=pd.Series(intensity[:,3])    #3
number11=pd.Series(number1[:,3])         #4

rr1=on1.corr(alpha1,method='pearson')
rr2=on1.corr(days1,method='pearson')
rr3=on1.corr(intensity1,method='pearson')
rr4=on1.corr(number11,method='pearson')


rrr=onn.corr(alpha1,method='pearson')
print('rrr','\n',rrr)
#print(rr1,rr2,rr3,rr4)

fig,ax=plt.subplots()

x=np.arange(0,19)
#plt.plot(x,on,lw=3,label='尼诺指数')
#plt.plot(x,alpha,lw=3,label='24h误差增长率')
#plt.plot(x,days,lw=3,label='可预测时间')
#plt.plot(x,intensity_scale,lw=3,label='强度')
#plt.plot(x,number_scale,lw=3,label='频数')
ax.bar(x,ninuo,color=np.where(ninuo>0,'orangered','deepskyblue'),width=0.5,label='Nino3.4')
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.tick_params(size=6,labelsize=23)
ax.set_xticks(np.arange(0,19))
ax.set_xticklabels(np.arange(2001,2020),rotation=45)
ax.set_yticks(np.arange(-2.5,2.5+0.1,0.5))

#ax.text(5,-2,'相关系数r=0.3275',horizontalalignment='center',rotation=0,
#        backgroundcolor='pink',fontsize=23,c='k',alpha=0.8)

plt.legend(loc='upper left',fontsize=23)

ax2=ax.twinx()
ax2.plot(x,alpha,c='mediumseagreen',label='24h误差增长率',lw=3)  #mediumseagreen
ax2.scatter(x,alpha,c='mediumseagreen',marker='d',s=37)
ax2.spines['top'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1)
ax2.tick_params(size=6,labelsize=23)
ax2.set_xticks(np.arange(0,19))
ax2.set_xticklabels(np.arange(2001,2020),rotation=45)
ax2.set_yticks(np.arange(0.8,2.2+0.1,0.2))
plt.legend(loc='upper left',bbox_to_anchor=(0.0,0.9),fontsize=23)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

plt.show()

print('厄尔尼诺',np.mean(parameter51[[1,3,5,8,14],1]*2))
print('拉尼娜',np.mean(parameter51[[6,9,15],1]*2))
print('中性年',np.mean(parameter51[[0,4,7,11,12,13,16,17,18],1]*2))

