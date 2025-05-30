import numpy as np
import math
a=np.loadtxt('D:/ZD/500.txt')
b=np.loadtxt('D:/ZD/850.txt')
rd=0.00976;p=3.14;R=287
omg=2*p/86400
for i in np.arange(0,29,1):
    al=a[i,2]/100  #纬度
    #bl=b[i,2]/100
    ah=a[i,3]     #位势高度
    bh=b[i,3]
    at=a[i,4]      #气温
    bt=b[i,4]
    awd=a[i,6]     #风向(a wind direction)
    bwd=b[i,6]
    aws=a[i,7]     #风速(a wind speed)
    bws=b[i,7]
    au=aws*math.cos(awd*p/180)  #高空u
    av=aws*math.sin(awd*p/180)
    bu=bws*math.cos(bwd*p/180)  #低空u
    bv=bws*math.sin(bwd*p/180)
    um=(au+bu)/2  #u中(u middle)
    vm=(av+bv)/2  #v中(v middle)
    ut=au-bu
    vt=av-bv
    r=-(at-bt)/(ah-bh)
    f=2*omg*math.sin(al*p/180)
    w=-(f*(um*vt-vm*ut)/R/math.log(850/500))/(rd-r)
    print(w)