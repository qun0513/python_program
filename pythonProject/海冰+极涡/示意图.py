import numpy as np
import math
import matplotlib.pyplot as plt
xx=np.linspace(0,2*np.pi,100)
climate=1.5*np.sin(xx)  #气侯波
bs=0.5*np.sin(xx)     #巴伦支海
os=0.5*np.sin(xx+np.pi)     #鄂霍茨克海
y=1.3*(climate+bs)
z=climate+os

fig,ax=plt.subplots()
plt.plot(xx,climate,c='r',ls='-',lw=2,label='1波气候波')
plt.plot(xx,bs,c='b',ls='-',lw=2,label='BS海冰强迫的波动异常')
plt.plot(xx,os,c='g',ls='-',lw=2,label='OS海冰强迫的波动异常')
plt.plot(xx,y,c='Magenta',ls=':',lw=2,label='1波气候波+BS海冰强迫的波动异常')
plt.plot(xx,z,c='orange',ls=':',lw=2,label='1波气候波+OS海冰强迫的波动异常')

ax.tick_params(size=6,labelsize=18)
ax.axhline(y=4.2,c='k',ls=':',lw=1.5)
ax.axvline(x=9,c='Cyan',ls=':',lw=1.5)
ax.axvline(x=9.2,c='Cyan',ls=':',lw=1.5)
ax.set_ylim(-3.0,6)
ax.set_xlim(0,10)
ax.set_title('季节、海域机制分析示意图',fontsize=23)
ax.text(1.6,4.6,'平流层',horizontalalignment='center',rotation=0,
        backgroundcolor='pink',fontsize=15,c='k',alpha=0.8)
ax.text(3.2,2.1,'波幅&能量',horizontalalignment='center',rotation=45,
        backgroundcolor='pink',fontsize=15,c='k',alpha=0.8)

plt.rcParams['font.sans-serif']=['KaiTi'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.legend(loc=7,fontsize=9,bbox_to_anchor=(0.9, 0.6))
plt.show()
