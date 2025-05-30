import numpy as np
import matplotlib.pyplot as plt

y=np.loadtxt('D:/download/qhtj/input.txt')              #输入数据
f=np.loadtxt('D:/download/qhtj/output.txt',skiprows=2)  #fortran程序输出的参数，并读入
x=np.arange(0,57)                                       #序列长度
Bs=f[1];lag1=f[4]                                       #slope、zstatistic

wt=np.zeros(56)
for i in np.arange(1,len(y)):
    wt[i-1]=(y[i]-lag1*y[i-1])/(1-lag1)
As=np.mean(wt)-Bs*np.mean(x+1)                          #截距

#判断显著性水平
if abs(f[3])>1.96:
    print('显著性水平为：5%\n置信度为95%')
sigtest='显著性水平为：5%\n置信度为95%'

#绘图
fig=plt.figure()
ax=fig.add_subplot()
ax.set_xlabel('Year',rotation=0,fontsize=25,labelpad=12)
ax.set_ylabel('Occurrence(%)',rotation=90,fontsize=25,labelpad=12)
ax.set_xticks(np.arange(0,57,10))
ax.set_xticklabels(['1960','1970','1980','1990','2000','2010'],fontsize=23)
ax.set_yticks(np.arange(0,25,5))
ax.tick_params(size=6,labelsize=23,width=1.5)
ax.set_title('WS2001',size=25,pad=12)

ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

  #在图片中显示显著性水平
ax.text(45,5,sigtest,fontsize=21,backgroundcolor='pink')

plt.plot(np.arange(0,57),y,ls='-',c='b',lw=3,label='初始序列')
plt.scatter(np.arange(0,57),y,c='b',marker='D',s=50)
plt.plot(np.arange(0,57),As+Bs*(x+1),ls='--',c='r',lw=3,label='WS2001')

plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False     #用来正常显示负号
plt.legend(loc='upper left',bbox_to_anchor=(0.05,0.95),frameon=False,fontsize=21)
plt.show()


#import scipy.io as sio

#python调用matlab-----------------------
#import matlab.engine
#eng = matlab.engine.start_matlab()
#eng.triarea(nargout=0)