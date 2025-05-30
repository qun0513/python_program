import numpy as np
import math
pi=3.14;r=6357000

s=0;s1=1
for i in np.arange(51,71,2):
    #方法一
    dx=2*pi*(r*math.cos(i*pi/180))*(1/360)
    dy=2*pi*r*(2/360)
    ss=dx*dy*75
    s=s+ss
    #方法二
    ss1=r**2*math.cos(i*pi/180)*(2*pi/180)*(1*pi/180)*75
    s1=s1+ss1
print(s,s1)