import numpy as np
import matplotlib.pyplot as plt
error=np.loadtxt('D:/毕业设计/error.txt')
error1=np.loadtxt('D:/毕业设计/error1.txt')

parameter51=np.array([[62.04347,0.73623,22.68116],
                      [56.38296,0.76915,22.75318],
                      [61.64586,0.52153,17.21529],
                      [47.65495,0.69649,21.05847],
                      [47.06239,0.75312,19.62496],
                      [41.86888,0.79344,18.94755],
                      [42.96438,0.66429,16.32146],
                      [44.13354,0.60890,15.04451],
                      [33.14947,0.85747,16.45979],
                      [45.17931,0.61195,13.05977],
                      [30.88777,0.84439,14.35511],
                      [32.18740,0.70937,12.27496],
                      [48.96165,0.59233,7.84659],
                      [27.10280,0.87352,10.36760],
                      [31.27481,0.81832,9.09160],
                      [28.84309,0.72287,7.61436],
                      [25.00396,0.72039,8.20158],
                      [27.43023,0.78702,6.77209],
                      [25.00396,0.72039,8.20158]])
#ysimulate=np.empty((16,5))
ytrue=np.empty((19,8))
for i in np.arange(0,19):
    for k in np.arange(0,8):
        if k<5:
            ytrue[i,k]=np.sqrt(parameter51[i,0]**2*np.exp(
                parameter51[i,1]*(k)*12/24))
        if k>=5:
            ytrue[i,k]=np.sqrt(parameter51[i,0]**2*np.exp(
                parameter51[i,1]*(k-2)*24/24))
print(ytrue[:,0])
print(ytrue[:,7])
print(ytrue[17,:])
print(ytrue[15,[2,4]])


#Figure_5
#绘图
fig=plt.figure()
ax=fig.add_subplot()

x21=np.arange(0,100)
x22=np.arange(38,51)
x23=np.arange(50,100)
yy=np.empty((100))
for i in np.arange(0,100):
    yy[i]=7.49/0.96196054**(i)
print(yy)

plt.plot(x21,yy,c='k',lw=3,ls='--')
plt.plot(x22,yy[38:51],c='b',lw=3)
plt.plot(x23,yy[50:100],c='r',lw=3)

#plt.scatter(0,yy[0],s=50)
plt.scatter(38,yy[38],s=50,c='b')
plt.scatter(50,yy[50],s=50,c='r')

xx=[55,52,54,47,47,44,45,46,39,46,36,38,48,34,37,35,34,34]
xy=[99,99,88,93,96,95,88,85,94,86,91,83,87,89,89,81,83,84]

for i in np.arange(0,19):
    if 12>i>=0:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                 np.linspace(xx[i], xy[i], 120)[47],
                 np.linspace(xx[i], xy[i], 120)[71],
                 np.linspace(xx[i], xy[i], 120)[95],
                 np.linspace(xx[i], xy[i], 120)[119],
                 ],error[i,2::],c='k',marker='*',s=63)
    if i==12:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='r', marker='*',s=63,label='2012')
    if i==13:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='limegreen', marker='*',s=63,label='2012')
    if i==14:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='orange', marker='*',s=63,label='2012')
    if i==15:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='g', marker='*',s=63,label='2016')
    #独立评估
    if i==16:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error1[i, 2::], c='b', marker='*',s=63,label='2017')
    if i==17:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error1[i, 2::], c='purple', marker='*',s=63,label='2018')
    if i==18:
        i = 16
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='pink', marker='*',s=63,label='2012')




print([np.linspace(35, 81, 120)[35],
                     np.linspace(35, 81, 120)[47],
                     np.linspace(35, 81, 120)[71],
                     np.linspace(35, 81, 120)[95],
                     np.linspace(35, 81, 120)[119],
                     ])

ax.set_xticks([np.linspace(38, 83, 120)[0],
               np.linspace(38, 83, 120)[23],
               #np.linspace(35, 81, 120)[35],
                     np.linspace(38, 83, 120)[47],
                     np.linspace(38, 83, 120)[71],
                     np.linspace(38, 83, 120)[95],
                     np.linspace(38, 83, 120)[119]])

ax.set_xticklabels(['0','1d','2d','3d','4d','5d'],c='b',fontsize=27)
ax.spines['right'].set_linewidth(0)
ax.spines['right'].set_color('None')
#ax.tick_params(bottom=True)
ax.axvline(x=38,c='b',ls=':',lw=2)
ax.axvline(x=50,c='r',ls=':',lw=2)

import matplotlib.ticker as mticker
#ax.xaxis.set_major_locator(mticker.FixedLocator([1,9,18,26,34,42]))
#ax.set_xticks([1,9,18,26,34,42])
#ax.tick_params(top=True,bottom=True,zorder=2)

ax_top=ax.twiny()
xticks=ax.get_xticks()
ax.set_xticks(xticks)
xlim=ax.get_xlim()
ax_top.set_xlim(xlim)

top_ticks=[2,14,26,38,50]
top_labels=['2045','2034','2023','2012','2001']
ax_top.set_xticks(top_ticks)
ax_top.set_xticklabels(top_labels)
ax_top.tick_params(labelsize=27,direction='out')
#ax.set_yticks(np.arange(0,351,25))
#ax.tick_params(labelsize=0,direction='in',length=1)
ax.set_yticks(np.arange(0,351,50))
ax.tick_params(labelsize=27,direction='in',length=5)

plt.legend(loc='upper left',bbox_to_anchor=(0.5,-0.37),ncol=3,
            frameon=False,fontsize=21)

ax.text(3,400,'NWP分析误差的逐年演变（年份）',fontsize=27)
ax.text(30,-50,'预测误差随提前期的演变(天数，以2012年为参考)',fontsize=27)

plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号
plt.show()