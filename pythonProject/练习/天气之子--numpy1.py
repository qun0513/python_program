import numpy as np
x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(x.ndim)  #纬度
print(x.shape) #形状
print(x.size)  #数量
print(x.dtype) #元素类型
print(x.itemsize) #每个元素的字节大小(4=32/8)

y=np.arange(15).reshape(3,5) #前开后闭
z=np.arange(0,15,5)
a=np.random.rand(2,3,4)  #rand随机样本位于[0,1]  random函数
b=np.random.randn(2,3,4) #从标准正态分布返回样本
#如果一个数组用来打印太大了，NumPy自动省略中间部分而只打印角落。
#禁用NumPy的这种行为并强制打印整个数组，你可以设置printoptions参数来更改打印选项。

print(a*b) #
#print(np.dot(a,b))
#print('xxxxxxxxxx')
#print(y.sum(axis=0))  #按列求成1行
#print(y.min(axis=1))
'''
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array(x).reshape(-1,1)+x  #broadcasting
print(x);print(y)
'''

#np.set_printoptions(threshold='nan')
#print(np.arange(90000).reshape(300,300))
import sys
np.set_printoptions(threshold=sys.maxsize)
#print(np.arange(10000).reshape(100,100))
#np.dot(),矩阵乘法
c=np.array([[1,2,3,4],[5,6,7,8]])  #array
#print(c.sum())  #.min/.max
#print(c.cumsum(axis=0)) #累加函数
#print(c.cumsum(axis=1)) #按行求成一列
#range()--list  np.arange()--array
c1=c[::-1] #c1,为多维，维度上倒序
#print(c1)
#print('xxx')
for d in c.flat:
    print(c,'\n')

