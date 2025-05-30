import math
import matplotlib.pyplot as plt
import numpy as np
import decimal
number=decimal.Decimal(0.00000002)


print(np.arange(-1,1,0.2))
yy=[]
for i in np.arange(-1,1,0.2):
    y=math.asin(i)
    print(y)
    yy=yy.append(y)
print(yy)
y=np.array(yy).reshape(1,10)

fig,ax=plt.subplots()
x=np.arange(-1,1,0.2)
ax.plot(x,y,ls='--',c='b',lw='1')
plt.show()
