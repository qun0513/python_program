#散点图
'''
常用关键字：x,y,s(控制散点的大小),c(控制散点颜色）,cmap(与c参数配合，修改等级颜色)
         alpha(指定散点透明度)，facecolor,edgecolor(强制指定颜色)
         linewidths/lw(指定散点边界粗细)，hatch(内部填充样式)
         vmin,vmax(裁剪色条上下限)
'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#中文正常显示
def sample_data():#编制实验数据
    x=range(1,21)#横坐标数据
    y=np.array([2,4,6,7,5,3,3,5,7,9,1115,10,8,4,7,8,3,2,5,7])#纵坐标数据
    data1=np.array([22,13,24,26,30,31,36,20,27,15,17,19,24,27,30,15,21,22,10,27])
    data2=np.array([120,132,143,151,109,149,125,119,120,158,171,101,106,108,126,149,127,151,143,102])
    return x,y,data1,data2
x,y,data1,data2=sample_data()#获得实验数据
fig=plt.figure(figsize=(2.5,2),dpi=500)
ax1=fig.add_axes([0.2,0.2,0.7,0.3])
ax2=fig.add_axes([0.2,0.6,0.7,0.3])
ax1.scatter(x,y,s=10,zorder=2,marker='*',c=data1,cmap='viridis',alpha=0.8,
            lw=1)
ax2.scatter(x,y,s=15,zorder=2,marker='+',c=data2,cmap='rainbow',alpha=0.5)
ax1.set(xlim=(0,21),ylim=(1,15))
ax2.set(xlim=(0,21),ylim=(1,15))
ax1.tick_params(axis='both',direction='out',length=3,width=0.5,labelsize=7)
ax2.tick_params(axis='both',direction='in',length=3,width=0.5,labelsize=7)
ax1.grid(alpha=0.75,ls=':')
ax2.grid(alpha=0.75,ls=':')
plt.tight_layout()
plt.show()

