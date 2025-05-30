
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#截取cmap
'''
cmap=mpl.cm.jet_r #获取色条
newcolors=cmap(np.linspace(0,1,256)) #分片操作
newcmap=ListedColormap(newcolors[125:]) #切片取舍
fig=plt.figure(figsize=(15,3),dpi=500)
ax1=fig.add_axes([0.1,0.3,0.7,0.15])
ax2=fig.add_axes([0.1,0.7,0.7,0.15])
norm =mpl.colors.Normalize(vmin=-10, vmax=16)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='jet'),
                 cax=ax1,#指定色条位置，具有最高优先级
                 orientation='horizontal',
                 extend='both')
fc2=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=newcmap),
                 cax=ax2,
                 orientation='horizontal',
                 extend='both')
for i in [fc1,fc2]:
    i.ax.tick_params(labelsize=3,width=0.5,length=1) #刻线的样式大小；刻线的粗细、长短
    i.outline.set_linewidth(0.5) #外框的线的粗细
plt.show()
'''
#拼接cmap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
cmap1=cmaps.spread_15lev_r
cmap2=cmaps.sunshine_diff_12lev
list_cmap1=cmap1(np.linspace(0,1,15))
list_cmap2=cmap2(np.linspace(0,1,12))
new_color_list=np.vstack((list_cmap1,list_cmap2))
new_cmap=ListedColormap(new_color_list,name='new_cmap ')  #
fig=plt.figure(figsize=(1.5,0.5),dpi=500)
ax1=fig.add_axes([0,0,1,0.33])
ax2=fig.add_axes([0,0.33,1,0.33])
ax3=fig.add_axes([0,0.66,1,0.33])
norm =mpl.colors.Normalize(vmin=0, vmax=10)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cmap1),cax=ax1,
                 orientation='horizontal',extend='both')
fc2=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=cmap2),cax=ax2,
                 orientation='horizontal',extend='both')
fc3=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=new_cmap),cax=ax3,
                 orientation='horizontal',extend='both')
plt.show()



'''import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap=mpl.cm.jet_r  #
cmap=cmap(np.linspace(0,1,256))
newcolormap=ListedColormap(cmap[125:])  #
norm=mpl.colors.Normalize(vmin=0,vmax=10)
fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(18,3),dpi=100)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='Reds'),
                                       cax=axes[0][0],
                                       extend='both',
                                       orientation='horizontal')
fc2=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=newcolormap),cax=axes[1][1],
                                       extend='both',
                                       orientation='vertical')
axes[0][0].tick_params(labelsize=3,width=0.5,length=1)
axes[0][0].tick_params(labelsize=5,width=1,length=1.5)
axes[1][1].tick_params(labelsize=3,width=0.5,length=1)
axes[1][1].tick_params(labelsize=5,width=1,length=1.5)
plt.show()
'''
