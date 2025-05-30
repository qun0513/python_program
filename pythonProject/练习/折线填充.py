import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d

x=np.arange(1,40,1)
y=np.array([0,1,3,2,4,5,7,6,21,15,
            19,23,27,14,10,5,4,7,
            5,8,3,9,11,22,29,31,
            34,27,40,52,33,20,19,
            16,14,60,55,54,66])
y2=x+5
xnew=np.linspace(1,39,200)
f1=interpolate.interp1d(x,y,kind='quadratic')
#kind连接数据点的线型,zero、nearest;linear、slinear；quadratic、cubic。
ynew=f1(xnew)
f2=interpolate.interp1d(x,y2,kind='quadratic')
ynew2=f2(xnew)
fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,12),dpi=100)

#辅助线axvline(竖直),axhline(水平)

#区间填充

ax.plot(xnew,ynew,lw=1,color='k',zorder=6)
ax.axvline(20,lw=0.5,color='k',zorder=6)
ax.axvline(30,lw=0.5,color='k',zorder=6)
slice1,slice2=100,153
ax.fill_between(xnew[slice1:slice2],
                y1=ynew[slice1:slice2],
                facecolor='tab:orange',zorder=5)


#双关系填充
'''
ax.plot(xnew,ynew,lw=1,color='k',
        zorder=6,label='ynew data')
ax.plot(xnew,ynew2,lw=1,color='#FF00FF',
        zorder=6,label='ynew2 data')
ax.fill_between(xnew, ynew, ynew2, where=ynew2>=ynew,
                facecolor='grey',interpolate=True,zorder=5)
ax.fill_between(xnew, ynew, ynew2, where=ynew2<=ynew,
                facecolor='orange',interpolate=True,zorder=5)
'''

#悬挂填充
'''
ax.plot(xnew,ynew,lw=1,color='k',zorder=6)
ax.axhline(40,lw=0.5,color='k',zorder=6)
ax.axhline(20,lw=0.5,color='k',zorder=6)
ax.fill_between(xnew,y1=ynew,y2=40,where=(ynew>=40),
                facecolor='tab:red',zorder=5)
ax.fill_between(xnew,y1=ynew,y2=20,where=(ynew<=20),
                facecolor='tab:blue',zorder=5)
'''
'''
#渐变色填充
points=np.array([xnew,ynew]).T.reshape(-1,1,2)
lw=1
cmap=plt.get_cmap('jet')
for i,c in zip(range(points.shape[0]),ynew):
    if i<range(points.shape[0])[-2]:
        ax.plot((points[i,0,0],points[i+1,0,0]),
                (points[i,0,1],points[i+1,0,1]),
                color='k',lw=lw)
        ax.fill_between((points[i,0,0],points[i+1,0,0]),
                        (points[i,0,1],points[i+1,0,1]),
                        facecolor=cmap((ynew[i]+ynew[i+1])/2/max(ynew)),
                        interpolate=True,zorder=5)
    else:
        ax.plot((points[i-1,0,0],points[i,0,0]),
                (points[i-1,0,1],points[i,0,1]),
                color='k',lw=lw)
        ax.fill_between((points[i-1,0,0],points[i,0,0]),
                        (points[i-1,0,1],points[i,0,1]),
                        facecolor=cmap(c/max(ynew)),interpolate=True,zorder=5)
'''
plt.show()
