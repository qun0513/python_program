import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from scipy.stats import pearsonr

pdoh=np.loadtxt("D:/decadal prediction/results/PDOindex_HadISST1950-2018.txt")
#print(pdo)

data1=pd.read_table('D:/decadal prediction/data/ersst.v5.pdo.dat.txt',sep=r'\s+',header=2)
pdo0=np.array(data1.values)
pdo1=pdo0[95:164,1:13]   # 1870-2018 : [15:164,1:13]
print(pdo1)
pdo2=pdo1.reshape(828)
'''
df = pd.DataFrame(pdo, columns=['Value'])
df['5Y_Moving_Average'] = df['Value'].rolling(window=60).mean()    #滑动平均
print(df['5Y_Moving_Average'])

pdo3=pd.Series(pdo2)
pc1=pd.Series(pdo)
r=pdo3.corr(pc1,method='pearson')                    #相关性
print('r','\n',r)
'''
pdoh1=np.empty((768))
pdo3=np.empty((768))
for i in np.arange(30, len(pdoh)-30):
    pdoh1[i-30]=np.mean(pdoh[i-30: i+30])
    pdo3[i-30]=np.mean(pdo2[i-30: i+30])
r,p=pearsonr(pdoh1,pdo3)
#print(pdoh1.shape)
#print(pdo3)
r=round(r,3)
#r=float(f"{r:.3f}")
#print(pdo1.shape)

fig=plt.figure()
ax=fig.add_subplot()

x=np.arange(0,828)
x1=np.arange(30,798)
ax.plot(x, pdoh, c='b', lw=0.6)
ax.plot(x1, pdoh1, c='k', lw=1.2)
ax.set_yticks(np.arange(-3,4,1))
ax.set_xticks([0,120,240,360,480,600,720,828])
ax.set_xlabel('year',size=30,labelpad=15)
ax.set_xticklabels([1950,1960,1970,1980,1990,2000,2010,2018])
#ax.tick_params(size=6, labelsize=24)
# 设置x轴的主刻度间隔
ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
# 设置x轴的次刻度间隔
ax.xaxis.set_minor_locator(ticker.MultipleLocator(12))
ax.tick_params(which='major', length=10, width=1.5, color='k')
ax.tick_params(which='minor', length=3, width=1, color='k')
ax.tick_params(size=6, labelsize=30)
ax.set_title('PDOindex_1950-2018',pad=30,size=30)

plt.axhline(y=0, color='black', linewidth=1,ls='--')
ax.annotate(str(r),xy=(0.8,0.95),xycoords='axes fraction',c='k',fontsize=25,ha='left',va='top')
#plt.axvline(x=216, color='g', linestyle='--', linewidth=2)
#plt.axvline(x=480, color='g', linestyle='--', linewidth=2)

y=0
ax.fill_between(x1,pdoh1,y,where=(pdoh1>y),interpolate=True,
               facecolor='r',alpha=0.8)   #tab:orange
ax.fill_between(x1,pdoh1,y,where=(pdoh1<y),interpolate=True,
               facecolor='b',alpha=0.8)

plt.subplots_adjust(left=0.080,
                    bottom=0.150,
                    right=0.950,
                    top=0.85,
                    wspace=0.1,      #子图间水平距离
                    hspace=0.15     #子图间垂直距离
                   )

plt.show()