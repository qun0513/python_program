import numpy as np
import matplotlib.pyplot as plt
#一、轴坐标名
'''
plt.xlabel(ylabel)或ax.set_xlabel(ylabel)
由于该命令完全是基于text模块的，所以他能使用text模块的关键字参数。下面列举了一些常用的设置关键字参数：
   关键字（fontsize:修改文本的字号； alpha:修改文本的透明度； backgroundcolor:背景颜色
         bbox:给标签加上多种样式的边框； color:标签文本的颜色； family:字体的样式，如黑体、仿宋
         ha、va:横向纵向的位置，如center、left； 文本旋转的角度； 
         position:标签的位置，（x，y）样式传入； loc: 控制标签的位置，‘top’，’left‘）
'''
#二、刻度
'''
#不过，某些时刻，我们一定要修改其位置，怎么做呢？有两个简单的办法：
#ax.xaxis.tick_top( )#或者
#ax.xaxis.set_ticks_position('top')
ax1.xaxis.tick_top()
ax2.xaxis.set_ticks_position('top')
ax3.yaxis.set_ticks_position('right')

接下来又有问题了，如果我想四边都要显示刻度怎么办呢？也有一个简单的命令：
ax1.tick_params(top=True,bottom=True,left=True,right=True)
那么，如何使刻度不显示呢？在设置刻度命令时传入空列表即可:
ax.set_xticks([ ])
ax.set_yticks([ ])

#关键字(matplotlib里有一个常用的调节刻度样式的命令——ax.tick_params( )。)
axis:指定要修改的轴，如'x','y','both'
which：要修改的主副刻度，如'major''minor''both'
direction：刻度的小齿的方向，如'in''out'
length：刻度小齿的长度
width：刻度小齿的宽度
color：刻度小齿的颜色
pad：刻度标签与刻度小齿的距离
labelsize：刻度标签字体的大小
labelcolor：刻度标签字体的颜色
rotation:旋转刻度标签

ax.set_xlim(10,20),也可颠倒其走向
ax.set_xticks()
ax.set_xticks([0.1,0.2,0.3,0.55])
'''
#刻度：关键的刻度格式环节，这个环节是颇为重要的，可以省去很多力气。
#     刻度格式生成器和刻度样式生成器是两个非常重要的模块。
#import matplotlib.ticker as mticker
'''
#首先使用刻度格式生成器，这个命令需要调用ax.xaxis.set
1.无刻度格式（Nulllocator）
ax.yaxis.set_major_locator(mticker.NullLocator())
2.等距离刻度格式（MultipleLocator）
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))  #刻度按这个来
3.抑制刻度数量命令（MaxNLocator）
ax.xaxis.set_major_locator(mticker.MaxNLocator(5))  #等分成几份
ax.yaxis.set_major_locator(mticker.MaxNLocator(2))
4.均匀刻度命令（LinearLocator）
ax.yaxis.set_major_locator(mticker.LinearLocator(10))  #该命令使刻度按划分数量均匀显示。
5.刻度固定（FixedLocator）
在该命令下，只会显示固定的刻度，即你指定要显示哪些刻度，其余的不显示。
ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(0,0.71,0.1)))
ax.yaxis.set_major_locator(mticker.FixedLocator([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]))
'''
'''
经纬度是我们最常用的刻度样式，目前在绘制地图时，有两种常用的方法——
一种是通过魔改matplotlib实现，一种是通过cartopy里的gridlines实现。
先讲魔改matplotlib的，这种实际上就是将子图坐标样式修改为经纬度，
这种以后还可以用ax.tick_params( )来修改刻度的艺术样式。

'''

#三、边框(轴)
'''
1.边框的隐现
a.对普通的子图  ax.axis('off')
b.对地图  ax.background_patch.set_visible(False) #使框线全部消失
         ax.outline_patch.set_visible(False)  %%%
c.对普通的子图，仅对某一边或几边去框线  ax.spines['top'].set_linewidth(0)#框线宽度为0，则在视觉效果上为无框线
                                 ax.spines['right'].set_color('None')#框线无颜色，在在视觉效果上为无框
d.对地图  ax1.outline_patch.set_visible(False)#关闭GeoAxes框线
         ax1.spines['left'].set_visible(True)#开启Axes框线
         ax1.spines['left'].set_linewidth(0)#框线宽度为0，则在视觉效果上为无框线
2.边框的粗细
3.坐标轴仅显示一部分  ax.spines['top'].set_linewidth(0)
                   ax.spines['right'].set_linewidth(0)
                   ax.spines['left'].set_bounds(-10,10)
                   ax.yaxis.set_major_locator(mticker.FixedLocator([-10,0,10]))
                  ax.spines['bottom'].set_bounds(0,50)
4.使坐标轴和子图内容物有一定的距离  ax.spines['left'].set_position(('outward',10))
                            ax.spines['bottom'].set_position(('outward',10))
5.使轴生产箭头
'''
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
def setup_axes(fig, rect):
    ax = axisartist.Subplot(fig, rect)
    fig.add_axes(ax)
    ax.set_ylim(-0.1, 1.5)
    ax.set_yticks([0, 1])
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(1, 0.5)
    ax.axis["x"].set_axisline_style("->", size=1.5)
    return ax
fig = plt.figure(figsize=(3, 2.5),dpi=200)
fig.subplots_adjust(top=0.8)
ax1 = setup_axes(fig, "111")
ax1.axis["x"].set_axis_direction("left")
plt.show()
'''
如何设置色条外框的粗？通过下面这个语句修改：
cb=fig.colorbar(......)
cb.outline.set_linewidth(0.1)
'''
