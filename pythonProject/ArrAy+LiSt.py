import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
obj1=Series([1,2,3,4],index=['a','b','c','d'])
obj2=Series({'e':1,'f':2,'g':3,'h':4})

print(obj1)
print(obj1['a'])
print(obj2)
data={'city':['lz','bj','sh'],'temp':[1,2,3]}  #字典
frame=DataFrame(data)
print(frame)
print(frame.temp)
#loc,iloc是标签运算符
# loc方法使用索引的名称来获取行或列
# iloc方法使用索引的位置来获取行或列


