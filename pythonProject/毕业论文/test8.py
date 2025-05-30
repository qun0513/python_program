import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
error=np.loadtxt('D:/毕业设计/error.txt')
sem=np.loadtxt('D:/毕业设计/sem.txt')
sem2=sem*2.18
w=np.empty((16,5))
for i in np.arange(0,16):
    for k in np.arange(2, 7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])


parameter4=np.array([64.2473877,0.93596054,0.7244406,0.35450621])

#模拟
ysimulate=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):

        if i>=12:
            j=j+1
            ysimulate[i,j-1]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+1)*24/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))
        else:
            if j<2:
                ysimulate[i,j]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+3)*12/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))
            if j>=2:
                ysimulate[i,j]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+1)*24/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))
#模拟+延伸
ysimulate1=np.empty((55,5))
for i in np.arange(0,55):
    for j in np.arange(0,5):
        if j<2:
            ysimulate1[i,j]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+3)*12/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))
        if j>=2:
            ysimulate1[i,j]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+1)*24/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))
print(ysimulate)

from scipy.optimize import curve_fit
def nihe(x,a,b):
    return a*np.exp(b*x/24)
simulate=np.empty((16,144))
x11=np.arange(0,144)

for i in np.arange(0,16):
    x=np.array([36,48,72,96,120])
    y=error[i,2::]                      #感知误差
    popt,pcov=curve_fit(nihe,x,y,maxfev = 1000000)  #p0=(1,1,1)
    #print(popt)
    #x=np.array(x)

    simulate[i] = (popt[0]*np.exp(popt[1]*x11/24))
#print(simulate)

#13年--205.40787962  15年——240.90722387

xvals1=np.linspace(0,3,72)
xvals=np.linspace(0,3,72)

#yinterp=np.interp(xvals,x,ysimulate[0,:])
#yvals=np.empty(16,120)
day=[]
day1=[]
error1=np.array([[136,133,170,205],
                 [85.8,131,168,240]])
# measured error
for i in np.arange(0,16):
    if i==12 or i==15:
        f1 = interp1d(np.arange(0, 4), error1[1, :], kind='linear')
        yvals1 = f1(xvals1)
        for j in np.arange(0, 72):
            if yvals1[j] < 181:
                d = j
        day1.append((d + 72) / 24)

    else:
        f1=interp1d(np.arange(0,4),error[i,3::],kind='linear')
        yvals=f1(xvals)
        #print(i,yvals)
        for j in np.arange(0,72):
            if yvals[j]<181:
                d=j
        day.append((d+48)/24)

# simulated error
da=[]
da1=[]
for i in np.arange(0,16):
    if i<12:
        f1 = interp1d(np.arange(0, 4), ysimulate[i, 1::], kind='linear')
        yv1 = f1(xvals1)
        for j in np.arange(0, 72):
            if yv1[j] < 181:
                d = j
        da1.append((d + 48) / 24)
    else:
        f1 = interp1d(np.arange(0, 4), ysimulate[i, 1::], kind='linear')
        yv = f1(xvals)
        for j in np.arange(0, 72):
            if yv[j] < 181:
                d = j
        da.append((d + 72) / 24)

yday=da1+da
print(yday)

#for i in np.arange(0,16):
#  plt.plot(np.arange(0,6),ysimulate1[i,:])
#  plt.plot(np.arange(0,6),ysimulate1[3,:],lw=2)

#days=np.array(day)
#print(days)
#print(day1)

days=np.array([2.625,2.83333333,4.33333333,3.54166667,3.41666667,3.5,
 4.33333333, 4.66666667,3.79166667, 4.5, 4.08333333,5.25,4.83333333,
 4.29166667, 4.16666667,5.125,5.7832,4.7716,6.1])

'''
#一元线性回归(最小二乘法)
Sxy=0;Sxx=0
y_bar=np.array(yday).mean()                                #   y 平均
xx=np.arange(0,16)
x_bar=xx.mean()                             #   x 平均
for i in np.arange(0,16):
    Sxy=Sxy+yday[i]*xx[i]
    Sxx=Sxx+xx[i]**2
k=(Sxy-16*y_bar*x_bar)/(Sxx-16*x_bar**2)               #
print(k)
bb=y_bar-k*x_bar                                       #
print(bb)

y=k*xx+bb
x1=np.arange(0,55)
y1=k*x1+bb
'''

#Figure_6
#绘图
fig,ax=plt.subplots()

x=np.arange(1,13)
k=0.1;b=3.3
y=k*x+b

x1=np.arange(1,56)
y1=k*x1+b

plt.scatter(x,days[0:12],marker='d',s=100,c='r')
plt.scatter(np.arange(13,20),days[12:19],marker='d',s=100,facecolor='white',
            edgecolors='r',linewidths=2)
plt.plot(x,y,c='k',lw=2)
plt.plot(x1,y1,ls='--',c='k',lw=2)

plt.vlines(x=7,ymin=0,ymax=4,ls=':',colors='k')
plt.vlines(x=17,ymin=0,ymax=5,ls=':',colors='k')
plt.vlines(x=27,ymin=0,ymax=6,ls=':',colors='k')
plt.vlines(x=47,ymin=0,ymax=8,ls=':',colors='k')
#ax.axhline(y=4,ls=':')
#ax.axhline(y=5,ls=':')
#ax.axhline(y=6,ls=':')
#ax.axhline(y=7,ls=':')

ax.set_yticks(np.arange(0,11))
ax.set_xticks(np.arange(1,55))

#ax.legend( loc='lower center',bbox_to_anchor=(0.5,-0.3),ncol=6,frameon=False,fontsize=16)
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_color('None')
ax.spines['bottom'].set_linewidth(0)
ax.spines['left'].set_color('None')
ax.set_ylabel('提前期（天数）',rotation=90,fontsize=23,labelpad=10)
ax.set_xticks(np.arange(1,56,2))
ax.set_xticklabels(np.arange(2001,2056,2),fontsize=23,rotation=45)
ax.set_yticks(np.arange(0,11,1))
ax.grid(axis='y',linestyle='-',lw=1)
ax.tick_params(size=6,labelsize=23)
#ax.text(-0.8,60,'(d)',fontsize=18)
#ax.set_title('2016',pad=10,fontsize=20)

plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号

plt.show()

