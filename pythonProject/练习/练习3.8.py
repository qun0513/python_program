import matplotlib.pyplot as plt
import numpy as np
fig,ax=plt.subplots()
x=np.array([1,2,3,4,5,6,7,8])
y=np.array([11,5,8,16,12,3,9,17])
ax.axhline(y=9,lw=1.5,ls=':',c='b')
ax.set_xlabel('Q',position=(0.5,1),c='tab:blue',rotation=45)
ax.plot(x,y,c='red',ls='-',lw=2)
ax.bar(x,y,color=np.where(y>11,'red','orange'),width=0.6)  #width:一般小于1
ax.set_title('W',c='m',rotation=45)
ax.tick_params(right=True,labeltop=True,rotation=45)
plt.show()


