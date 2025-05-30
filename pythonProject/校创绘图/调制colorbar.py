import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

#棕+蓝  *2
cmap1=mpl.cm.YlOrBr
cmap2=mpl.cm.Blues
list_cmap1=cmap1(np.linspace(0,1,6))
list_cmap2=cmap2(np.linspace(0,1,6))

cmap7=mpl.cm.bwr
list_cmap7=cmap7(np.linspace(0,1,12))
list_cmap7=ListedColormap(list_cmap7[0:6],name='list_cmap7')
list_cmap7=list_cmap7(np.linspace(0,1,6))

'''
list_cmap4=ListedColormap(list_cmap2[0:1])
list_cmap3=ListedColormap(list_cmap1[6:7])
list_cmap3=np.vstack((list_cmap3,list_cmap4))
list_cmap5=ListedColormap(list_cmap1[4:6])
list_cmap3=np.vstack((list_cmap5,list_cmap3))
list_cmap6=ListedColormap(list_cmap1[0:4])
list_cmap3=np.vstack((list_cmap6,list_cmap3))
list_cmap3=ListedColormap(list_cmap3[::],name='list_cmap3')
list_cmap3=list_cmap3(np.linspace(0,1,8))
'''


new_color_list1=np.vstack((list_cmap1,list_cmap2))
new_cmap1=ListedColormap(new_color_list1,name='new_cmap2')  #colormap1

new_color_list2=np.vstack((list_cmap7,list_cmap1))
new_cmap2=ListedColormap(new_color_list2,name='new_cmap2')  #colormap2
###############################################################################
cmap4=mpl.cm.Reds
cmap5=mpl.cm.Blues_r
list_cmap4=cmap4(np.linspace(0,1,6))
list_cmap5=cmap5(np.linspace(0,1,6))

new_color_list1=np.vstack((list_cmap5,list_cmap4))
new_cmap3=ListedColormap(new_color_list1,name='new_cmap3')  #colormap1


fig=plt.figure(figsize=(15,5),dpi=200)
ax1=fig.add_axes([0.2,0.3,0.6,0.06])
ax2=fig.add_axes([0.2,0.5,0.6,0.06])
norm =mpl.colors.Normalize(vmin=-6, vmax=6)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=new_cmap3),cax=ax1,
                 orientation='horizontal',extend='both')
fc2=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=new_cmap2),cax=ax2,
                 orientation='horizontal',extend='both')
fc1.ax.set_title('配色1')
fc2.ax.set_title('配色2')
plt.show()


#
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmaps

plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

cmap1=mpl.cm.Reds_r
cmap2=mpl.cm.bwr_r
cmap3=mpl.cm.seismic

list_cmap1=cmap1(np.linspace(0,1,6))
list_cmap2=cmap2(np.linspace(0,1,12))
list_cmap2=ListedColormap(list_cmap2[6:12],name='list_cmap7')
list_cmap2=list_cmap2(np.linspace(0,1,6))
list_cmap3=cmap3(np.linspace(0,1,12))

new_color_list1=np.vstack((list_cmap1,list_cmap2))
new_cmap1=ListedColormap(new_color_list1,name='new_cmap2')  #colormap1

#new_color_list2=np.vstack((list_cmap1,list_cmap7))
#new_cmap2=ListedColormap(new_color_list2,name='new_cmap2')  #colormap2

new_cmap3=ListedColormap(list_cmap3,name='new_cmap3')       #colormap3

fig=plt.figure(figsize=(15,5),dpi=200)
ax1=fig.add_axes([0.2,0.3,0.6,0.06])
#ax2=fig.add_axes([0.2,0.6,0.6,0.06])
norm =mpl.colors.Normalize(vmin=-1500, vmax=1500)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=new_cmap3),cax=ax1,
                 orientation='horizontal',extend='both')
#fc2=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
#                 cmap=new_cmap2),cax=ax2,
#                 orientation='horizontal',extend='both')
fc1.ax.set_title('配色1')
#fc2.ax.set_title('配色2')
plt.show()
'''

#蓝+棕
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmaps
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

cmap1=mpl.cm.YlOrBr
cmap2=mpl.cm.Blues_r
list_cmap1=cmap1(np.linspace(0,1,6))
list_cmap2=cmap2(np.linspace(0,1,6))

new_color_list1=np.vstack((list_cmap2,list_cmap1))
new_cmap1=ListedColormap(new_color_list1,name='new_cmap2')  #colormap1

fig=plt.figure(figsize=(15,5),dpi=200)
ax1=fig.add_axes([0.2,0.3,0.6,0.06])
norm =mpl.colors.Normalize(vmin=-6, vmax=6)
fc1=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,
                 cmap=new_cmap1),cax=ax1,
                 orientation='horizontal',extend='both')
fc1.ax.set_title('配色1')
plt.show()

'''