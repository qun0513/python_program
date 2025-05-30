#折线图
'''
import numpy as np
import matplotlib.pyplot as plt
#plt.plot();ax.plot()
#plot()命令是在内部传入x轴、y轴数据，两者的数据不能长度不一，然后电脑自动在笛卡尔坐标系中按顺序连接这些点。

fig,ax=plt.subplots(figsize=(15,12),dpi=500)
#可行
plt.plot([1],[1],marker='*')#列表单独一个点
plt.plot(1,1,marker='*')#数值单独一个点
plt.plot((1),(1),marker='*')#元组单独一个点
plt.show()


x=[1,2,3,4,5]
y1=[1,2,3,4,5]
y2=[2,3,4,5,6]
y3=[3,4,5,6,7]
y4=[4,5,6,7,8]
y5=[5,6,7,8,9]
#参数
plt.plot(x,y1,color='k',ls=':',lw='0.5',marker='+',markeredgecolor='m',markersize=2)
plt.plot(x,y2,color='b',ls='--',lw='1',marker='*',markerfacecolor='r')
plt.plot(x,y3,color='r',ls='-.',lw='5',marker='_',markeredgewidth='0.5')#marker不能为—（减号）
plt.plot(x,y4,marker='o',fillstyle='none')#none、top、left
plt.plot(x,y5,marker='^',label='QY',visible=True)  #visible(True\False)
plt.legend()
plt.show()
#zoeder:指定绘图层次（高层次会盖住低层次0,1\...）
#visible:是否显示该线条(True\False)

'''


#堆积折线图
'''
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(3,3),dpi=500)#添加画布
x=[1,2,3,4,5]
y1=[1,2,3,4,5]
y2=[1,2,3,4,5]
y3=[1,2,3,4,5]
colors=['tab:blue','tab:red','tab:orange']  #指定填充颜色，从低到高
plt.stackplot(x,y1,y2,y3,colors=colors)  #(多个自变量)
plt.legend(loc='upper left')  # best\ upper right\ upper left\lower left
                               # \lower right\ right\center left\center right
                               # \lower center\ upper center\center
plt.tight_layout()# tight_layout会自动调整子图参数，使之填充整个图像区域。
                  # 这是个实验特性，可能在一些情况下不工作。
                  # 它仅仅检查坐标轴标签、刻度标签以及标题的部分。
#plt.show()
'''



#填充（fill）
'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(3,3),dpi=500)
x=[1,2,3,4,5,6,7,8,9,10,11]
y=[1,2,3,4,5,2,-1,-5,-7,-2,1]
plt.plot(x,y,c='tab:blue',marker='o',markersize=3)
plt.fill(x,y,color='tab:green',alpha=0.5) #alpha:设置像素（透明度）?  y-y
plt.show()
'''

'''
#填充（fill_between）
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(3,3),dpi=500)
ax=fig.add_axes([0,0,1,1])
x=np.array([1,2,3,4,5,6,7,8,9,10,11])
y1=np.array([1,2,3,4,5,2,-1,-5,-7,-2,1])
y2=np.array([3,4,5,3,1,-1,-4,-6,-3,2,1])
ax.plot(x,y1,ls=':',c='k',marker='o',fillstyle='none')
ax.plot(x,y2,ls='-.',c='k')
ax.fill_between(x,y1,y2,where=(y1>y2),interpolate=True,
               facecolor='tab:orange',alpha=0.8)
ax.fill_between(x,y1,y2,where=(y2>y1),interpolate=True,
               facecolor='tab:blue',alpha=0.8)
plt.show()
'''

#折线图的多坐标轴
import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
x=np.array([1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20])
y1=np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1])
y2=np.array([950,960,970,980,990,1000,1107,1108,1109,
             1110,1109,1108,1107,1106,1200,1000,980,960,940])
y3=np.array([0.11,0.22,0.43,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.9,
             0.8,0.7,0.6,0.5,0.4,0.3,0.2,1])
             
'''         
ax.plot(x,y1,ls=':')
ax.plot(x,y2,ls='-.')
ax.plot(x,y3,ls='--')
'''

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

ax.plot(x,y1,c='tab:blue',ls=':',marker='o')
ax.plot(x,y3,c='tab:green',ls='--',marker='^')
ax2=ax.twinx()
ax2.plot(x,y2,c='tab:red',ls='--',marker='*')
line1,=ax.plot(x,y1,c='tab:blue',ls=':',marker='o')
line3,=ax.plot(x,y3,c='tab:green',ls='--',marker='^')
ax2=ax.twinx()
line2,=ax2.plot(x,y2,c='tab:red',ls='--',marker='*')
ax.legend([line1,line2,line3],['介于1-10','介于900-1200','介于0.1-1'],loc='upper left')
plt.show()

