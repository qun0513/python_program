import numpy as np
import math
a=np.loadtxt('D:/ZD/500.txt')
p=3.14;l0=1.11137
Bij=[]
Bik=[]
#将输出结果写入文件
import sys
sys.stdout=open('D:/ZD/客观分析——多项式法.log',mode='w',encoding='utf-8') #.log 或 .txt

#一、求矩阵B#########################################################################
for i in np.arange(0,29,1):
    lon=a[i,1]/100  #经度
    lat=a[i,2]/100  #纬度
    z=a[i,3]        #位势高度
    x=l0*math.cos(((lat+25)/2)*p/180)*(lon-115)
    y=l0*(lat-25)
    for p in [0,1,2]:
        for q in np.arange(0,p+1):  # [ , )  前闭后开所以是p+1
            b=x**(p-q)*y**q
            Bij.append(b)        #在列表末尾不断添加新的数
            Bik.append(b)        #同上
    Bik.append(z)                #同上，且Bik=Bij+z
Bij=np.array(Bij).reshape(29,6)  #将列表变为数组，再变形为（29，6）
Bik=np.array(Bik).reshape(29,7)  #同上，且Bik即为所求B矩阵
print('矩阵B','\n',Bik)            #输出：B矩阵

#二、求矩阵C （6，29）*（29，7）=（6，7）################################################
 #1、先求Bij的转置矩阵，（6，29）
Bji=np.empty((6,29))
for i in np.arange(0,29):
    for j in np.arange(0,6):
        Bji[j,i]=Bij[i,j]

 #2、Bji*Bik，（6，29）*（29，7），即为C矩阵（6，7）
C=[]
for j in np.arange(0,6,1):
  for k in np.arange(0,7,1):
    Cjk=0
    for i in np.arange(0,29,1):
          cjk=Bji[j,i]*Bik[i,k]  #对应位置相乘，共29次
          Cjk=Cjk+cjk            #累加上述29个数，作为C的一个Cjk
    C.append(Cjk)                #在列表末尾不断添加新的Cjk
C=np.array(C).reshape(6,7)       #将列表转为数组，再变形为（6，7）
print('矩阵C','\n',C)              #输出：C矩阵

#三、求系数a(k)######################################################################
  #1、先化成上三角矩阵 + 1列
a=np.empty((6,7))
for i in np.arange(0,6):
    for j in np.arange(0,7):
        a[i,j]=C[i,j]/C[i,i]             #将第i列的对角线元素化为1之后的 新的行
    for k in np.arange(i+1,6):
        for l in np.arange(0,7):
            a[k,l]=C[k,l]-C[k,i]*a[i,l]  #之后的每一行都减去：上面求出行的倍数（倍数为C[k,i]）
    for m in np.arange(0,6):
        for n in np.arange(0,7):
            C[m,n]=a[m,n]                #将这一遍求出的a赋给C，后续在 新C 的基础上继续循环

  #2、再化成对角矩阵 + 1列 （最后一列即为a（k））
b=np.empty((6,7))
for i in np.arange(4,-1,-1):             #倒着运算，从下往上
    for j in np.arange(6,-1,-1):
        b[5,j]=a[5,j]                    #最后一行不再进行变化，直接赋给b
    for k in np.arange(i,-1,-1):
        for l in np.arange(0,7):
            b[k,l]=a[k,l]-a[k,i+1]*b[i+1,l]#上面的每一行都减去：第i行的倍数（倍数为a（k，i+1））
    for m in np.arange(0,6):
        for n in np.arange(0,7):
            a[m,n]=b[m,n]                #将这一遍求出的b赋给a，后续在新a的基础上继续循环
print('最后一列为系数a（k）','\n',b)   #输出b（单位对角矩阵+系数a（k）一列）