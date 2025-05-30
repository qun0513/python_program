import matplotlib.pyplot as plt
import numpy as np
fig,ax=plt.subplots(figsize=(50,40),dpi=100)
x=[1,2,3,4]
y=[1,3,1,4]
ax.set_xticks([1,3,4])
ax.set_xlim(0,8)
ax.set_ylim(0,8)
#ax.set_xticks(np.arange(0,5,0.5))

ax.set_xlabel('Z',fontsize=25,labelpad=10)
ax.set_ylabel('Q',fontsize=25,labelpad=100)
ax.set_title('ZQ',fontsize=30,pad=100)
#ax.gridlines(ls='--')
ax.tick_params(labelsize=20)
#ax.plot(x,y,lw=2,ls='-.',c='tomato')
plt.subplot(236)
plt.bar(x,y)
plt.subplot(224)
plt.scatter(x,y)
plt.show()
