import math
k=0.7156;a=6370;p=3.14159
'''
#将输出结果写入文件
import sys
sys.stdout=open('D:/ZD/放大系数m.log',mode='w',encoding='utf-8') #.log 或 .txt
'''
#验证标准维度的m
l0=(a*math.sin(p/3))/k
m=(k*l0)/(a*math.sin(p/3))
print(m,'验证')
#求(5,6)的m
for y in [0,1,2,3,4]:
    ly=l0-(4-y)*100
    for x in [0,1,2,3,4,5]:
        lx=x*100
        l=(lx**2+ly**2)**(1/2)
        t=(l/l0)**(1/k)*math.tan(p/6)  # t 为 tan（）
        s=(2*t)/(1+t**2)               # s 为 sin（）
        m=(k*l)/(a*s)
        print(m)