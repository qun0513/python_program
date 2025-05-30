import numpy as np
import matplotlib.pyplot as plt
n=5
m=4
o=3
np.random.seed(1)
x=np.random.randint(1,10,size=(3,3,3))
np.random.seed(1)
y=np.random.randint(10,20,size=(3,3,3))
#np.random.randn(m,n)，两个及以上参数，矩阵；
#无参数，返回一个浮点数；一个参数，返回秩为1的数组.
#randn返回正态分布的数值；rand返回[0,1)的数值；
#randint,numpy.random.randint(low,high=None,size=None,dtype)
#  生成在半开半闭区间[low,high)上离散均匀分布的整数值;若high=None，则取值区间变为[0,low)
print(x,y)
plt.scatter(x,y)
plt.show()
