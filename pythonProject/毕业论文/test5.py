import numpy as np
error=np.loadtxt('D:/毕业设计/error.txt')
error1=np.loadtxt('D:/毕业设计/error1.txt')
sem=np.loadtxt('D:/毕业设计/sem.txt')
sem2=sem*2.18
w=np.empty((16,5))
for i in np.arange(0,16):
    for k in np.arange(2, 7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])


parameter4=np.array([57.23389881,0.95465204,0.72303738,0.35315367])
#parameter4=np.array([64.2473877,0.93596054,0.7244406,0.35450621])
#模拟
ysimulate=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
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


#Figure_4
#绘图
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
x1=np.arange(0,12)
x2=np.arange(0,55)
plt.scatter(x1,error[0:12,2],c='r',marker='s',label='36h感知误差')     #36h
plt.plot(x1,ysimulate[0:12,0],c='r',label='36h估计误差')
plt.plot(x2,ysimulate1[:,0],c='r',ls='--',label='36h外延')
#独立评估
plt.scatter(np.arange(12,19),error[12:19,2],c='white',marker='s',
            linewidths=2,facecolor='white',edgecolors='r')

plt.scatter(x1,error[0:12,4],c='limegreen',marker='.',s=200,label='72h感知误差')     #72h
plt.plot(x1,ysimulate[0:12,2],c='limegreen',label='72h估计误差')
plt.plot(x2,ysimulate1[:,2],c='limegreen',ls='--',label='72h外延')
#独立评估
plt.scatter(np.arange(12,19),error[12:19,4],marker='.',s=200,
            linewidths=2,facecolor='white',edgecolors='limegreen')

plt.scatter(x1,error[0:12,6],c='royalblue',marker='d',label='120h感知误差')     #120h
plt.plot(x1,ysimulate[0:12,4],c='royalblue',label='120h估计误差')
plt.plot(x2,ysimulate1[:,4],c='royalblue',ls='--',label='120h外延')
#独立评估
plt.scatter(np.arange(12,19),error[12:19,6],marker='d',
            linewidths=2,facecolor='white',edgecolors='royalblue')

handles,labels = ax.get_legend_handles_labels()

#改变legend的默认顺序
handles=[handles[6],handles[0],handles[1],handles[7],handles[2],handles[3],
         handles[8],handles[4],handles[5]]
labels = [labels[6],labels[0],labels[1],labels[7],labels[2],labels[3],
          labels[8],labels[4],labels[5]]
ax.legend(handles,labels, loc='lower center',bbox_to_anchor=(0.5,-0.37),ncol=3,
            frameon=False,fontsize=21)

ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_color('None')
ax.spines['bottom'].set_linewidth(0)
ax.spines['left'].set_color('None')
ax.set_ylabel('感知误差（海里）',rotation=90,fontsize=23,labelpad=18)
ax.set_xticks(np.arange(0,55,2))
ax.set_xticklabels(np.arange(2001,2056,2),
                    rotation=90,fontsize=16)
ax.set_yticks([0,50,100,150,200,250,300,350,400])
ax.grid(axis='y',linestyle='-',lw=1)
ax.tick_params(size=6,labelsize=23)
#ax.text(-6.0,400,'(a)',fontsize=18)
#ax.set_title('2001',pad=10,fontsize=20)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong SimSun KaiTi
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.show()