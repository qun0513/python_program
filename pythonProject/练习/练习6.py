#Pandas 数据读写 处理时间序列 分组
import pandas as pd
from pandas import Series
from pandas import DataFrame
from datetime import datetime,timedelta
a=Series([[1,2],[3,4]],index=['a','b'])#数据结构，通常为表格的一列
print(a.values)#a.index
print(a['b'])
b=Series({'a':1,'b':2,'c':3})
print(b)
c=[[1,2,3],[4,5,6]]
d=['x','y']
e=Series(c,d)
print(e)
e.index=('u','v')
print(e)
data={'city':['lanzhou','beijing','shanghai'],'temp':[1,2,3]}
f=DataFrame(data)#表格型的数据结构
print(f)
g=DataFrame(data,columns=['temp','city','windsped'],index=['x','y','z'])
print(g)#索引、列、值
#DataFrame loc选取和iloc选取：名称、位置
print(g.drop(['x'],axis=0))#删除行或列
#数据读写
'''frame=pd.read_csv('')
frame1=pd.to_csv('')#如果不需要输出索引，可以加上index=False'''
#时间序列  datatime time calenda
h=datetime.now();h1=h.year;h2=h.month
print(h,h1,h2)
i=h.timetuple().tm_yday
print(i)
j=DataFrame({'城市':['lanzhou','tianshui','baiying'],'省份':['gansu','gansu','gansu'],'国家':['china','china','china']})
k=j['省份'].str.cat(j['城市'],sep='-').str.cat(j['国家'],sep='--')
print(j,k)#cat连接，replace替换
'''l=h.dt.year
print(l)'''
