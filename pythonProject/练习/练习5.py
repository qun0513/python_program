#Numpy提供了两种基本的对象：ndarray、ufufnc
#ndarray是储存单一数据类型的多维数组
#ufunc是一种能够对数组进行处理的函数
import numpy as np
from numpy import random as nr

a=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
print(a.shape)
b=a.reshape((4,3))
print(b)
c=np.arange(0,60,10).reshape(-1,1)+np.arange(0,6)  #(-1,1)变为一列，（1，-1）变为一行
print(c)
d=nr.rand(4,3)
print(d)
e=np.sum(d,axis=0)
print(e)
f=np.min(d)#可以加轴，axis=1，0
print(f)
g=np.arange(0,10)
h=np.arange(0,10)
print(g)
i=np.polyfit(g,h,1)#一元线性回归
print(i)
j,k=np.meshgrid(g,h)#用于快速生成网格坐标纸
print(k)

