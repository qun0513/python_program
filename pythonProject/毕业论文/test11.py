import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

oni=pd.read_table('D:/毕业设计/尼诺指数.txt',sep=r'\s+',header=0)
#b=pd.read_table('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt',
#                sep=r'\s+',header=7)

y=oni['Nino34'].groupby(oni['Year'])
nindex=y.mean()
parameter51=np.array([[62.04347,0.73623,22.68116],
                      [56.38296,0.76915,22.75318],
                      [61.64586,0.52153,17.21529],
                      [47.65495,0.69649,21.05847],
                      [47.06239,0.75312,19.62496],#5
                      [41.86888,0.79344,18.94755],
                      [42.96438,0.66429,16.32146],
                      [44.13354,0.60890,15.04451],
                      [33.14947,0.85747,16.45979],
                      [45.17931,0.61195,13.05977],#10
                      [30.88777,0.84439,14.35511],
                      [32.18740,0.70937,12.27496],
                      [48.96165,0.59233,7.84659],
                      [27.10280,0.87352,10.36760],
                      [31.27481,0.81832,9.09160],
                      [28.84309,0.72287,7.61436]])
#print(np.array(nindex)[19:36])
#print(nindex)
#print(nindex[19:36])
on=np.array(nindex)[19:35]
alpha=parameter51[:,1]*2
print(on)
print(alpha)

on1=pd.Series(on)
alpha1=pd.Series(alpha)
rr=on1.corr(alpha1,method='pearson')
print(rr)

#归一化
from sklearn.preprocessing import scale
on_scale=scale(on)
alpha_scale=scale(alpha)
#print(on_scale)
#print(alpha_scale)
fig,ax=plt.subplots()

x=np.arange(0,16)
plt.plot(x,on_scale,lw=3,label='尼诺指数')
plt.plot(x,alpha_scale,lw=3,label='24h误差增长率')
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.tick_params(size=6,labelsize=23)
ax.set_xticks(np.arange(0,16))
ax.set_xticklabels(np.arange(2001,2017),rotation=45)
ax.set_yticks(np.arange(-3,3+0.1,0.5))

ax.text(5,-2,'相关系数r=0.3275',horizontalalignment='center',rotation=0,
        backgroundcolor='pink',fontsize=23,c='k',alpha=0.8)

plt.legend(fontsize=23)

plt.rcParams['font.sans-serif']=['KaiTi'] #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

plt.show()