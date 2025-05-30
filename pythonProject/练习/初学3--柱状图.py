#柱状图
#ax.bar(),ax.barh()  前者为竖直，后者为水平
'''
常用参数：color、facecolor（修改柱体颜色）
        width（调整柱体宽度），bottom（调整柱状图底部的原始高度）
        edgecolor（边框颜色），linewidth（调整边框宽度）
        xerr，yerr（误差线），ecolor（误差线颜色），capsize（误差线帽长度）
        align（edge或center，柱体位置），hatch（柱体内网格线填充样式）
        alpha（透明度），log（是否开启对数坐标轴）
'''


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif']=['SimHei']#中文
plt.rcParams['axes.unicode_minus']=False#负号
'''
def sample_data():#拟造数据
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    y=np.array([11,14,18,2,16,4,15,3,19,17,4,11,1,18,14,3,13,19,2,3])
    return x,y
x,y=sample_data()#获得数据
fig=plt.figure(figsize=(4,2),dpi=500)#添加画布
ax1=fig.add_axes([0,0,1,0.4])#添加子图
ax2=fig.add_axes([0,0.5,1,0.4])
ax1.bar(x,y,color='tab:green',bottom=2,width=0.2)
ax2.bar(x,y,color='tab:blue',bottom=1,width=0.5)
#############下面是针对图表刻度等细节的修改##############
ax1.set_ylim(0,20)
ax1.tick_params(axis='both',which='both',direction='in')
ax1.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(2))
ax2.set_ylim(0,20)
ax2.tick_params(axis='both',which='both',direction='in')
ax2.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(2))
plt.tight_layout()
plt.show()
'''



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif']=['SimHei']#中文
plt.rcParams['axes.unicode_minus']=False#负号
def sample_data():#拟造数据
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
               41,42,43,44,45,46,47,48,49,50])
    x2=np.array([1,2,3,4,5])
    y1=np.array([11,14,18,2,16,-4,-15,3,19,17,-4,11,20,18,14,3,13,19,2,3,16,15,12,
                10,4,-2,-4,-11,-18,-17,-3,5,9,15,17,8,9,16,14,11,-8,5,17,6,-9,-5,11,
                -7,-2,10])
    y2=np.array([14,4,6,14,18,15,19,21,-9,-6,-4,-2,-1,-3,-5,-6,-8,1,11,9,15,13,17,
                19,10,4,6,9,2,12,11,8,-7,-9,5,5,4,15,12,13,-9,2,1,4,11,15,17,8,11,16])
    y3=np.array([1120,1230,1190,1340,1590,1180,1390,1520,1690,1370,1210,1550,1555,1320,1221,1421,
                1321,1532,1432,1222,1243,1543,1121,1672,1389,1567,1678,1224,1521,1790,1810,1146,
                1356,1455,1789,1567,1234,1120,1675,1879,1800,1567,1235,1786,1346,1345,1789,1435,1567,1333])
    y4=np.array([24,27,29,35,33,21])
    y5=np.array([14,24,27,19,30])
    return x,x2,y1,y2,y3,y4,y5
x,x2,y1,y2,y3,y4,y5=sample_data()#获得数据
fig=plt.figure(figsize=(5,4),dpi=600)#添加画布等
ax1=fig.add_axes([0,0,1,0.4])
ax2=fig.add_axes([0,0.5,1,0.4])
ax1.bar(x,y1,width=0.3,color=np.where(y1>3,'tomato','tab:blue'))  #限值变色where
ax2.bar(x,y2,width=0.2,color='k')
ax1.set_ylim(-20,25)
ax1.tick_params(axis='both',which='both',direction='in')
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(5))
ax1.set_xlabel('时刻')
ax1.set_ylabel('数值')
ax2.set_ylim(-20,25)
ax2.tick_params(axis='both',which='both',direction='in')
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(1))
ax2.set_xlabel('时刻')
ax2.set_ylabel('数值')
plt.title('柱状图')
plt.show()


'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif']=['SimHei']#中文
plt.rcParams['axes.unicode_minus']=False#负号
def sample_data():#拟造数据
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
               41,42,43,44,45,46,47,48,49,50])
    x2=np.array([1,2,3,4,5])
    y1=np.array([11,14,18,2,16,-4,-15,3,19,17,-4,11,20,18,14,3,13,19,2,3,16,15,12,
                10,4,-2,-4,-11,-18,-17,-3,5,9,15,17,8,9,16,14,11,-8,5,17,6,-9,-5,11,
                -7,-2,10])
    y2=np.array([14,4,6,14,18,15,19,21,-9,-6,-4,-2,-1,-3,-5,-6,-8,1,11,9,15,13,17,
                19,10,4,6,9,2,12,11,8,-7,-9,5,5,4,15,12,13,-9,2,1,4,11,15,17,8,11,16])
    y3=np.array([1120,1230,1190,1340,1590,1180,1390,1520,1690,1370,1210,1550,1555,1320,1221,1421,
                1321,1532,1432,1222,1243,1543,1121,1672,1389,1567,1678,1224,1521,1790,1810,1146,
                1356,1455,1789,1567,1234,1120,1675,1879,1800,1567,1235,1786,1346,1345,1789,1435,1567,1333])
    y4=np.array([24,27,29,35,33,21])
    y5=np.array([14,24,27,19,30])
    return x,x2,y1,y2,y3,y4,y5
x,x2,y1,y2,y3,y4,y5=sample_data()#获得数据
fig=plt.figure(figsize=(50,20),dpi=100)
ax=fig.add_axes([0.1,0.1,0.8,0.8])
bar=ax.bar(x,y3,color=np.where(y3>1500,'tomato','tab:blue'))

min_bar=y3.argmin()
max_bar=y3.argmax()
bar[min_bar].set_color('orange')
bar[max_bar].set_color('r')
bar[max_bar].set_hatch("*")

#Pandas 里面的 idxmin 、idxmax函数
# 与Numpy中 argmax、argmin 用法大致相同，
# 这些函数将返回第一次出现的最小/最大值的索引。

ax.axhline(y=1500,c='k',ls=':',lw=1)
ax.axvline(x=5,c='m',lw=0.5,ls=':')
ax.set_ylim(1000,2000)
#ax.tick_params(size=20,labelsize=80)
ax.tick_params(labelsize=18,axis='both',which='both',direction='in')
ax.yaxis.set_minor_locator(mticker.MultipleLocator(100))
ax.text(45,1650,'QY',horizontalalignment='center',rotation=270,
        backgroundcolor='b',fontsize=50
        #fontsize：字体大小;color：str or tuple, 设置字体颜色
        #backgroundcolor：字体背景颜色;
        #horizontalalignment(ha)：设置垂直对齐方式，可选参数：left,right,center
        #verticalalignment(va)：设置水平对齐方式 ，可选参数 ： ‘center’ , ‘top’ , ‘bottom’ ,‘baseline’
        #rotation(旋转角度)：可选参数为:vertical,horizontal 也可以为数字
        #alpha：透明度，参数值0至1之间
        )
ax.set_title('限值变色',size=20,pad=10,family='FangSong')
#plt.savefig('a',bbox_inches='tight')
plt.legend(bar,'Q')
plt.show()
'''

'''
#如何并列多个数据(并列柱状图)手动挪
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif']=['SimHei']#中文
plt.rcParams['axes.unicode_minus']=False#负号
def sample_data():#拟造数据
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
               41,42,43,44,45,46,47,48,49,50])
    x2=np.array([1,2,3,4,5])
    y1=np.array([11,14,18,2,16,-4,-15,3,19,17,-4,11,20,18,14,3,13,19,2,3,16,15,12,
                10,4,-2,-4,-11,-18,-17,-3,5,9,15,17,8,9,16,14,11,-8,5,17,6,-9,-5,11,
                -7,-2,10])
    y2=np.array([14,4,6,14,18,15,19,21,-9,-6,-4,-2,-1,-3,-5,-6,-8,1,11,9,15,13,17,
                19,10,4,6,9,2,12,11,8,-7,-9,5,5,4,15,12,13,-9,2,1,4,11,15,17,8,11,16])
    y3=np.array([1120,1230,1190,1340,1590,1180,1390,1520,1690,1370,1210,1550,1555,1320,1221,1421,
                1321,1532,1432,1222,1243,1543,1121,1672,1389,1567,1678,1224,1521,1790,1810,1146,
                1356,1455,1789,1567,1234,1120,1675,1879,1800,1567,1235,1786,1346,1345,1789,1435,1567,1333])
    y4=np.array([24,27,29,35,33])
    y5=np.array([14,24,27,19,30])
    return x,x2,y1,y2,y3,y4,y5
x,x2,y1,y2,y3,y4,y5=sample_data()#获得数据
fig=plt.figure(figsize=(3,1.5),dpi=600)#添加画布等
ax=fig.add_axes([0.1,0.1,0.7,0.7])
bar1=ax.bar(x2-0.22,y4,width=0.45)
bar2=ax.bar(x2+0.23,y5,width=0.45)
ax.set_title('并列柱状图',fontsize=6)
ax.set(xlim=(0,6),ylim=(0,40))
ax.tick_params(labelsize=4,axis='both',which='both',direction='in',size=6)
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
#自动添加数字
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    size=5
                    )
autolabel(bar1)
autolabel(bar2)
plt.show()
#三个及以上建议用公式挪
'''
'''
#如何在柱体头部标值
def sample_data():#拟造数据
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
               41,42,43,44,45,46,47,48,49,50])
    x2=np.array([1,2,3,4,5])
    y1=np.array([11,14,18,2,16,-4,-15,3,19,17,-4,11,20,18,14,3,13,19,2,3,16,15,12,
                10,4,-2,-4,-11,-18,-17,-3,5,9,15,17,8,9,16,14,11,-8,5,17,6,-9,-5,11,
                -7,-2,10])
    y2=np.array([14,4,6,14,18,15,19,21,-9,-6,-4,-2,-1,-3,-5,-6,-8,1,11,9,15,13,17,
                19,10,4,6,9,2,12,11,8,-7,-9,5,5,4,15,12,13,-9,2,1,4,11,15,17,8,11,16])
    y3=np.array([1120,1230,1190,1340,1590,1180,1390,1520,1690,1370,1210,1550,1555,1320,1221,1421,
                1321,1532,1432,1222,1243,1543,1121,1672,1389,1567,1678,1224,1521,1790,1810,1146,
                1356,1455,1789,1567,1234,1120,1675,1879,1800,1567,1235,1786,1346,1345,1789,1435,1567,1333])
    y4=np.array([24,27,29,35,33])
    y5=np.array([14,24,27,19,30])
    return x,x2,y1,y2,y3,y4,y5
x,x2,y1,y2,y3,y4,y5=sample_data()#获得数据
fig=plt.figure(figsize=(15,10),dpi=250)#添加画布等
ax=fig.add_axes([0.1,0.15,0.7,0.7])
bar1=ax.bar(x2-0.22,y4,width=0.45)
bar2=ax.bar(x2+0.23,y5,width=0.45)
ax.set_title('并列柱状图',fontsize=10)
ax.set(xlim=(0,6),ylim=(0,40))
ax.tick_params(axis='both',which='both',direction='in')
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(bar1)
autolabel(bar2)
plt.show()

'''
#黑白刊物投稿
'''
#堆积柱状图(这个图的绘制要用到bottom参数，将第二个bar的下界设定为第一个bar高度值)
def sample_data():#拟造数据
    x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
               41,42,43,44,45,46,47,48,49,50])
    x2=np.array([1,2,3,4,5,6])
    y1=np.array([11,14,18,2,16,-4,-15,3,19,17,-4,11,20,18,14,3,13,19,2,3,16,15,12,
                10,4,-2,-4,-11,-18,-17,-3,5,9,15,17,8,9,16,14,11,-8,5,17,6,-9,-5,11,
                -7,-2,10])
    y2=np.array([14,4,6,14,18,15,19,21,-9,-6,-4,-2,-1,-3,-5,-6,-8,1,11,9,15,13,17,
                19,10,4,6,9,2,12,11,8,-7,-9,5,5,4,15,12,13,-9,2,1,4,11,15,17,8,11,16])
    y3=np.array([1120,1230,1190,1340,1590,1180,1390,1520,1690,1370,1210,1550,1555,1320,1221,1421,
                1321,1532,1432,1222,1243,1543,1121,1672,1389,1567,1678,1224,1521,1790,1810,1146,
                1356,1455,1789,1567,1234,1120,1675,1879,1800,1567,1235,1786,1346,1345,1789,1435,1567,1333])
    y4=np.array([24,27,29,35,33,21])
    y5=np.array([14,24,27,19,30,22])
    return x,x2,y1,y2,y3,y4,y5
x,x2,y1,y2,y3,y4,y5=sample_data()#获得数据
fig=plt.figure(figsize=(15,10),dpi=300)
ax=fig.add_axes([0.15,0.2,0.75,0.7])
bar1=ax.bar(x2,y4,width=0.6)
bar2=ax.bar(x2,y5,width=0.6,bottom=y4)

ax.set_title('堆积柱状图',fontsize=10)
ax.set(xlim=(0,6.5),ylim=(0,80))
ax.tick_params(labelsize=8,axis='both',which='both',direction='in')
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))


ax.legend([bar1,bar2],['红色柱形','蓝色柱形'])
plt.show()
'''


