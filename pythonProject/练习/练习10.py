import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
fig,ax=plt.subplots()
x=np.arange(1,40,1)  #上限为40
y=np.array([0,1,3,2,4,5,7,6,21,15,
            19,23,27,14,10,5,4,7,
            5,8,3,9,11,22,29,31,
            34,27,40,52,33,20,19,
            16,14,60,55,54,66])
x1=np.linspace(1,39,160)  #新的x
                          #上限为39
                          # A value in x_new is above the interpolation range.
f1=interpolate.interp1d(x,y,kind='quadratic')  #定义插值函数
y1=f1(x1)                 #新的x引起新的y
ax.plot(x1,y1)
plt.show()