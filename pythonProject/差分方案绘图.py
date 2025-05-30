import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#frame=pd.read_excel('D:/FORTRAN/差分方案/差分方案1.xlsx')
#frame.to_csv('D:/FORTRAN/差分方案/frame1.csv')
a=pd.read_csv('D:/FORTRAN/差分方案/frame1.csv',header=None)
print(a)
b=a.loc[:,1]
c1=np.array(b)
d=a.loc[:,2]
c2=np.array(d)
e=a.loc[:,3]
c3=np.array(e)
print(c1)
print(c2)
print(c3)

fig,ax=plt.subplots(figsize=(50,45),dpi=98)
x=np.arange(1,1002)

ax.set_xlabel('时间差分',fontsize=16)
ax.set_ylabel('动能',fontsize=16)
#ax.set_ylim(150,301)
ax.set_yticks([200,220,240,260,280])
ax.set_title('动能随时间差分变化趋势图',fontsize=20,pad=20)
ax.axhline(y=202,c='k',ls=':',lw=2)
plt.rcParams['font.sans-serif']=['FangSong'] #用来正常显示中文标签 SimHei
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
ax.plot(x,c1,lw=2,c='r')
ax.plot(x,c2,lw=2,c='b')
ax.plot(x,c3,lw=2,c='m')
p1,=ax.plot(x,c1,lw=2,c='r')
p2,=ax.plot(x,c2,lw=2,c='b')
p3,=ax.plot(x,c3,lw=2,c='m')
plt.legend([p1,p2,p3],['固定边界条件','周期边界条件','相邻边界条件'],loc='upper left')
plt.show()