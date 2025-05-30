import numpy as np
import matplotlib.pyplot as plt
#num_list=[1.5,0.6,7.8,6]
#plt.bar(range(len(num_list)),num_list,color=['r','b','g','y'])
#plt.show()
n=12
X=np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.0,1.0,n)
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
for X,Y1 in zip(X,Y1):
    plt.text(X+0.4,Y1=0.05)     #加数字
plt.show()
'''
mean,sigma=0,1
x=mean+sigma*np.random.randn(10000)
plt.hist(x,50)
plt.show()
'''
'''
import pandas as pd
plt.plot(X,Y1)
plt.fill_between(X,1,Y2,color='b',alpha=0.25)
plt.plot(X,Y1-1,color='r',)
plt.show()
'''



