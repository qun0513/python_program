from matplotlib import pyplot as plt
import numpy as np
'''
fig=plt.figure(figsize=(2.4,1.6),dpi=80)
#figure为最高一级对象
#add_axes
#subplots
ax1=fig.add_axes([0.1,0.1,0.3,0.2],facecolor='y')
ax2=fig.add_axes([0.6,0.1,0.3,0.5],facecolor='r')
#plt.show()
'''
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(2.4,1.6),dpi=200)
x=np.arange(0,10)
y=np.arange(0,10)

ax[0].set_title('x-y')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_xticks(np.arange(0,10,2))
ax[0].set_yticks(np.arange(0,10,2))
ax[0].grid('''which='minor' ''',linestyle=':',color='b')
ax[0].plot(x,y,marker='+',linestyle='-',color='r')

plt.show()



