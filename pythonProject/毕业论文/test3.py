import numpy as np
error=np.loadtxt('D:/毕业设计/error.txt')
sem=np.loadtxt('D:/毕业设计/sem.txt')
sem2=sem*2.18
w=np.empty((16,5))
for i in np.arange(0,16):
    for k in np.arange(2, 7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])

#独立评估
error1=np.loadtxt('D:/毕业设计/error1.txt')
sem1=np.loadtxt('D:/毕业设计/sem1.txt')
sem21=sem1*2.18
w1=np.empty((18,5))
for i in np.arange(0,18):
    for k in np.arange(2, 7):
        w1[i,k-2]=sem21[i,k]/sum(sem21[i,2::])

#coe_last=np.zeros((3))
#coe_start=np.zeros((3))
#coe_int=np.zeros((3))

#coe_start[0]=15;coe_start[1]=0.5;coe_start[2]=0           #初值
#coe_int[0]=0.1;coe_int[1]=0.01;coe_int[2]=0.001         #间隔

sim=np.empty((18,5))
sim_err=np.empty((18,5))
xbest=np.zeros((18,3))
xbest1=np.zeros((18,3))
xbest2=np.zeros((18,3))
#simmin=1e20

xx=[[62,65],[52,56],[54,58],[43,46],[40,46],[40,46],[40,46],[43,46],
    [27,30],[42,46],[29,31],[28,30],[45,48],[22,25],[29,31],[29,31]]
xy=[[23,25],[20,22],[21,23],[21,23],[17,20],[16,18],[16,18],[14,15],
    [17,19],[14,15],[15,17],[15,17],[11,13],[12,14],[9,11],[7,9]]

x=np.array([36,48,72,96,120])
#y=error[0,2::]
coe_last=np.array([60,1,25])
#print(coe_last[0])
xi = np.array([12, 12, 24, 24, 24])
y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
step=0.8
amp=0.1

noptimum=np.zeros((18))

#模型试验------------------------------------------------------------------
for i in np.arange(0,18):
  coe_last[0]=76-3*i;coe_last[1]=0.9;coe_last[2]=24 -i
  simmin = 1e20

  #y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
  for j in np.arange(0,1000):  #250000
    wg=np.array([0.001,0.0001,0.0004])

    coe_add=coe_last+wg
    y_add=(coe_add[0]**2*np.exp(coe_add[1]*(x)/24)+coe_add[2]**2)**0.5
    error_add=np.std(error1[i,2::]-y_add)

    coe_sub=coe_last-wg
    y_sub=(coe_sub[0]**2*np.exp(coe_sub[1]*(x)/24)+coe_sub[2]**2)**0.5
    error_sub=np.std(error1[i,2::]-y_sub)

    error_e=(error_add-error_sub)*step*wg
    coe_last=coe_last-error_e

    q=0;fmax=0
    for n in np.arange(0,5):
        if n < 2:
            sim[i,n]=((coe_last[0]**2*np.exp(coe_last[1]*(n+3)*12/24)
                       +coe_last[2]**2-error1[i,n+2]**2)**2)/w1[i,n]
            sim_err[i,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(n+3)*12/24)
                                         -error1[i,n+2])
        if n >= 2:
            sim[i,n]=((coe_last[0]**2*np.exp(coe_last[1]*(n+1)*24/24)
                                    +coe_last[2]**2-error1[i,n+2]**2)**2)/w1[i,n]
            sim_err[i,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(n+1)*24/24)
                                         -error1[i,n+2])
        if sim_err[i, n] < sem21[i, n]:
            q = q + 1
        if sim[i, n] > fmax:
            fmax = sim[i, n]
        if simmin > fmax and q == 5:
            simmin=fmax
            xbest[i,0]=coe_last[0]
            xbest[i,1]=coe_last[1]
            xbest[i,2]=coe_last[2]
            #print(xbest)
            print('hello,world!',i)
            noptimum[i]=noptimum[i]+1
        '''
        if q==5:
            xbest1[i,0]=coe_last[0]
            xbest1[i,1]=coe_last[1]
            xbest1[i,2]=coe_last[2]
            #print(xbest)
            print('Hello,world!',i)
            coe1_max = coe_last[1]
            if coe_last[1]>1:
                xbest2[i, 0] = coe_last[0]
                xbest2[i, 1] = coe_last[1]
                xbest2[i, 2] = coe_last[2]
        '''
  #print(i)
  print(i)
print('xbest','\n',xbest)
#print('xbest1','\n',xbest1)
#print('xbest2','\n',xbest2)
print(noptimum)
#print(error_e)


#51(48)参数--误差模型
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
                      [31.27481,0.81832,9.09160],
                      [28.84309,0.72287,7.61436]])
ysimulate=np.empty((16,5))
ytrue=np.empty((16,8))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate[i,j]=np.sqrt(parameter51[i,0]**2*np.exp(parameter51[i,1]*(j+3)*12/24)
                                   +parameter51[i,2]**2)
            sim_err[i,j]=abs(parameter51[i,0]*np.exp(0.5*parameter51[i,1]*(j+3)*12/24)
                                         -error[i,j+2])
        if j>=2:
            ysimulate[i,j]=np.sqrt(parameter51[i,0]**2*np.exp(parameter51[i,1]*(j+1)*24/24)
                                   +parameter51[i,2]**2)
            sim_err[i,j]=abs(parameter51[i,0]*np.exp(0.5*parameter51[i,1]*(j+1)*24/24)
                                         -error[i,j+2])
    for k in np.arange(0,8):
        if k<5:
            ytrue[i,k]=np.sqrt(parameter51[i,0]**2*np.exp(parameter51[i,1]*(k)*12/24))
        if k>=5:
            ytrue[i,k]=np.sqrt(parameter51[i,0]**2*np.exp(parameter51[i,1]*(k-2)*24/24))

print('ysimulate','\n',ysimulate)
print('ytrue','\n',ytrue)

with open('D:/毕业设计/simulate_perceive.txt','w') as outfile:
    np.savetxt(outfile,ysimulate)
with open('D:/毕业设计/estimate_true.txt','w') as outfile:
    np.savetxt(outfile,ytrue)

'''
#figure_2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
fig=plt.figure()

ax1=fig.add_subplot(221)  #--------------------------------1
#label1=['36h','48h','72h','96h','120h']
cc=['orange','royalblue','forestgreen','deepskyblue','orangered']
x1=np.arange(3,8)
x2=np.arange(0,8)
y1=error[0,2::]
y2=ysimulate[0,:]
y3=ytrue[0,:]
plt.plot(x1,y1,c='orange',lw=3,label='测量的感知误差')
plt.plot(x1,y2,c='forestgreen',lw=3,ls='--',label='拟合的感知误差',zorder=3)
plt.plot(x2,y3,c='deepskyblue',lw=3,label='估计的真实误差')
ax1.legend( loc='lower center',bbox_to_anchor=(0.5,-0.55),ncol=2,
            frameon=False,fontsize=20)
#ax1.axis('off')
ax1.spines['top'].set_linewidth(0)
ax1.spines['right'].set_color('None')
ax1.spines['bottom'].set_linewidth(0)
ax1.spines['left'].set_color('None')
ax1.set_ylabel('误差（海里）',rotation=90,fontsize=23,labelpad=16)
ax1.set_xticks(np.arange(0,8))
ax1.set_xticklabels(['0h','12h','24h','36h','48h','72h','96h','120h'],
                    rotation=0,fontsize=23)
ax1.set_yticks([0,100,200,300,400])
ax1.grid(axis='y',linestyle='-',lw=1)
ax1.tick_params(size=6,labelsize=23)
ax1.text(-1.7,400,'(a)',fontsize=23)
ax1.set_title('2001',pad=10,fontsize=23)


ax2=fig.add_subplot(222)  #--------------------------------2
#cc=['orange','royalblue','forestgreen']

x1=np.arange(3,8)
x2=np.arange(0,8)
y1=error[11,2::]
y2=ysimulate[11,:]
y3=ytrue[11,:]
plt.plot(x1,y1,c='orange',lw=3,label='测量的感知误差')
plt.plot(x1,y2,c='forestgreen',lw=3,ls='--',label='拟合的感知误差',zorder=3)
plt.plot(x2,y3,c='deepskyblue',lw=3,label='估计的真实误差')


ax2.legend( loc='lower center',bbox_to_anchor=(0.5,-0.55),ncol=2,
            frameon=False,fontsize=20)
ax2.spines['top'].set_linewidth(0)
ax2.spines['right'].set_color('None')
ax2.spines['bottom'].set_linewidth(0)
ax2.spines['left'].set_color('None')
ax2.set_ylabel('误差（海里）',rotation=90,fontsize=23,labelpad=16)
ax2.set_xticks(np.arange(0,8))
ax2.set_xticklabels(['0h','12h','24h','36h','48h','72h','96h','120h'],
                    rotation=0,fontsize=23)
ax2.set_yticks(np.arange(0,250,50))
ax2.grid(axis='y',linestyle='-',lw=1)
ax2.tick_params(size=6,labelsize=23)
ax2.text(-1.7,200,'(b)',fontsize=23)
ax2.set_title('2012',pad=10,fontsize=23)


ax3=fig.add_subplot(223)  #--------------------------------3
#cc=['orange','royalblue','forestgreen','deepskyblue','orangered']

x3=np.arange(0,5)
y1=sem2[0,2::]
y2=sim_err[0,:]
plt.plot(x3,y1,c='deepskyblue',lw=3,label='SEM')
plt.plot(x3,y2,c='orange',lw=3,label='拟合误差的绝对值')

ax3.legend( loc='lower center',bbox_to_anchor=(0.5,-0.37),
            ncol=5,frameon=False,fontsize=20)
ax3.spines['top'].set_linewidth(0)
ax3.spines['right'].set_color('None')
ax3.spines['bottom'].set_linewidth(0)
ax3.spines['left'].set_color('None')
ax3.set_ylabel('拟合误差（海里）',rotation=90,fontsize=23,labelpad=16)
ax3.set_xticks(np.arange(0,5))
ax3.set_xticklabels(['36h','48h','72h','96h','120h'],
                    rotation=0,fontsize=23)
ax3.set_yticks(np.arange(0,150,20))
ax3.grid(axis='y',linestyle='-',lw=1)
ax3.tick_params(size=6,labelsize=23)
ax3.text(-0.95,140,'(c)',fontsize=23)
ax3.set_title('2001',pad=10,fontsize=23)


ax4=fig.add_subplot(224)  #--------------------------------4
#cc=['wheat','lightskyblue','bisque','palegreen',
#    'silver','cornflowerblue','sandybrown','limegreen',
#    'gray','deepskyblue','chocolate','lightcoral',
#    'darkorange','orangered','dodgerblue','forestgreen']
x3=np.arange(0,5)
y1=sem2[11,2::]
y2=sim_err[11,:]
plt.plot(x3,y1,c='deepskyblue',lw=3,label='SEM')
plt.plot(x3,y2,c='orange',lw=3,label='拟合误差的绝对值')

ax4.legend( loc='lower center',bbox_to_anchor=(0.5,-0.37),
            ncol=6,frameon=False,fontsize=20)
ax4.spines['top'].set_linewidth(0)
ax4.spines['right'].set_color('None')
ax4.spines['bottom'].set_linewidth(0)
ax4.spines['left'].set_color('None')
ax4.set_ylabel('拟合误差（海里）',rotation=90,fontsize=23,labelpad=16)
ax4.set_xticks(np.arange(0,5))
ax4.set_xticklabels(['36h','48h','72h','96h','120h'],fontsize=23)
ax4.set_yticks(np.arange(0,70,10))
ax4.grid(axis='y',linestyle='-',lw=1)
ax4.tick_params(size=6,labelsize=23)
ax4.text(-0.90,60,'(d)',fontsize=23)
ax4.set_title('2012',pad=10,fontsize=23)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong SimSun KaiTi
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.show()
'''


#figure_3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
fig=plt.figure()

ax1=fig.add_subplot(131)  #--------------------------------1
#label1=['36h','48h','72h','96h','120h']
cc=['orange','royalblue','forestgreen','deepskyblue','orangered']
x1=np.arange(0,12)
y1=parameter51[0:12,0]
plt.plot(x1,y1,c='deepskyblue',lw=3)

ax1.legend( loc='lower center',bbox_to_anchor=(0.5,-0.5),ncol=1,
            frameon=False,fontsize=23)
#ax1.axis('off')
ax1.spines['top'].set_linewidth(0)
ax1.spines['right'].set_color('None')
ax1.spines['bottom'].set_linewidth(0)
ax1.spines['left'].set_color('None')
ax1.set_ylabel('初始误差（海里）',rotation=90,fontsize=23,labelpad=10)
ax1.set_xticks(np.arange(0,12,2))
ax1.set_xticklabels(np.arange(2001,2013,2),
                    rotation=45,fontsize=23)
ax1.set_yticks(np.arange(0,71,10))
ax1.grid(axis='y',linestyle='-',lw=1)
ax1.tick_params(size=6,labelsize=23)
ax1.text(-3.2,70,'(a)',fontsize=23)
#ax1.set_title('2001',pad=10,fontsize=20)


ax2=fig.add_subplot(132)  #--------------------------------2
#cc=['orange','royalblue','forestgreen']

x2=np.arange(0,12)
y2=parameter51[0:12,2]
plt.plot(x2,y2,c='deepskyblue',lw=3)

ax2.legend( loc='lower center',bbox_to_anchor=(0.5,-0.5),ncol=1,
            frameon=False,fontsize=23)
ax2.spines['top'].set_linewidth(0)
ax2.spines['right'].set_color('None')
ax2.spines['bottom'].set_linewidth(0)
ax2.spines['left'].set_color('None')
ax2.set_ylabel('最佳路径误差（海里）',rotation=90,fontsize=23,labelpad=10)
ax2.set_xticks(np.arange(0,12,2))
ax2.set_xticklabels(np.arange(2001,2013,2),
                    rotation=45,fontsize=23)
ax2.set_yticks(np.arange(0,31,5))
ax2.grid(axis='y',linestyle='-',lw=1)
ax2.tick_params(size=6,labelsize=23)
ax2.text(-3.2,30,'(b)',fontsize=23)
#ax2.set_title('2016',pad=10,fontsize=20)


ax3=fig.add_subplot(133)  #--------------------------------3
#cc=['orange','royalblue','forestgreen','deepskyblue','orangered']

x3=np.arange(0,12)
y3=parameter51[0:12,1]*2
plt.plot(x3,y3,c='deepskyblue',lw=3)

ax3.legend( loc='lower center',bbox_to_anchor=(0.5,-0.3),ncol=5,
            frameon=False,fontsize=23)
ax3.spines['top'].set_linewidth(0)
ax3.spines['right'].set_color('None')
ax3.spines['bottom'].set_linewidth(0)
ax3.spines['left'].set_color('None')
ax3.set_ylabel('24h误差增长率（≥36h）',rotation=90,fontsize=23,labelpad=10)
ax3.set_xticks(np.arange(0,12,2))
ax3.set_xticklabels(np.arange(2001,2013,2),
                    rotation=45,fontsize=23)
ax3.set_yticks(np.arange(0.6,2.21,0.2))
ax3.grid(axis='y',linestyle='-',lw=1)
ax3.tick_params(size=6,labelsize=23)
ax3.text(-3.2,2.2,'(c)',fontsize=23)
#ax3.set_title('2001',pad=10,fontsize=20)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong SimSun KaiTi
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.show()




'''
ysimulate1=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate1[i,j]=np.sqrt(xbest1[i,0]**2*np.exp(xbest1[i,1]*(j+3)*12/24)
                                   +xbest1[i,2]**2)
        if j>=2:
            ysimulate1[i,j]=np.sqrt(xbest1[i,0]**2*np.exp(xbest1[i,1]*(j+1)*24/24)
                                   +xbest1[i,2]**2)
    #ysimulate=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(x)/24)+xbest[i,2]**2)
print('ysimulate1','\n',ysimulate1)
'''


