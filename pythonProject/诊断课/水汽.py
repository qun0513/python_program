import numpy as np
import os
import math

#数据读取过程###################################################################
path_t = "D:/ZD/DATA/678/T" #文件夹目录
files_t= os.listdir(path_t) #得到文件夹下的所有文件名称
path_td="D:/ZD/DATA/678/TD"
files_td= os.listdir(path_td)
path_uv="D:/ZD/DATA/678/UV"
files_uv= os.listdir(path_uv)

def BR(files_t,path_t):  #(Batch Reading,批量读取)
    data_list = []
    for file in files_t:  # 遍历文件夹
        position = path_t + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
        # print (position)
        with open(position, "r", encoding='utf-8') as f:  # 打开文件
            data0 = f.readlines()  # 读取文件
        for line in data0:
            line = line.strip('\n')
            data_split = line.split()
            temp = list(map(float, data_split))
            data_list.append(temp)
    data1 = np.array(data_list)
    print(data1.shape)
    data = data1.reshape(7,18,10)  # 最终所用数组维度
    return data
T=BR(files_t,path_t)
TD=BR(files_td,path_td)
#UV=BR(files_uv,path_uv)   (11,36,10)
#print(T)
#print(TD)

data_list1 = []
for file in files_uv:  # 遍历文件夹
    position = path_uv + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
    #print (position)
    with open(position, "r", encoding='utf-8') as f:  # 打开文件
        data2 = f.readlines()  # 读取文件
    for line in data2:
        line = line.strip('\n')
        data_split = line.split()
        temp = list(map(float, data_split))  #数字+分隔
        data_list1.append(temp)
data3 = np.array(data_list1)
#print(data3)
#print(data3.shape)
UV = data3.reshape(7,36,10)
#print(UV)

#计算过程#########################################################################
g=9.8;d=400                      #常数
e=np.empty((7,18,10))            #水汽压
q=np.empty((7,18,10))            #水汽
Q=np.empty((7,18,10))            #水汽的平流项
D=np.empty((7,18,10))            #相关散度项
fm=np.empty((18,10))             #放大系数m

R=np.empty((7,18,10))            #水汽通量
r=np.zeros((18,10))
p=np.array([200,250,300,400,500,700,850])  #压强 pressure

#求解全过程
for k in np.arange(0,7):
  for i in np.arange(0,18):
    for j in np.arange(0,10):
        #求水汽q
        if T[k,i,j]>-15:
            e[k,i,j]=6.1078*math.exp(17.26*TD[k,i,j]/(273.16+TD[k,i,j]-35.86))
        if -40<=T[k,i,j]<-15:
            e0=6.1078*math.exp(17.26*TD[k,i,j]/(273.16+TD[k,i,j]-35.86))
            e1=6.1078*math.exp(21.8746*TD[k,i,j]/(273.16+TD[k,i,j]-7.66))
            e[k,i,j]=0.022*((80+2*TD[k,i,j])*e0+(30+2*TD[k,i,j])*e1)
        if T[k,i,j]<-40:
            e[k,i,j]=6.1078*math.exp(21.8746*TD[k,i,j]/(273.16+TD[k,i,j]-7.66))
        q[k,i,j]=0.622*e[k,i,j]/p[k]
  #求水汽平流项  Q
  kk = 0.7156;a = 6370;pi = 3.14159
  l0 = (a*math.sin(pi/3))/kk
  ll=np.empty((18,10))
  t=np.empty((18,10))
  s=np.empty(((18,10)))
  for m in np.arange(0,18):
      ly=l0-(17-m)*400
      for n in np.arange(0,10):
        #先求放大系数m
        lx=n*400
        ll[m,n]=(lx**2+ly**2)**(1/2)
        t[m,n]=(ll[m,n]/l0)**(1/kk)*math.tan(pi/6)  # t 为 tan（）
        s[m,n]=(2*t[m,n])/(1+t[m,n]**2)  # s 为 sin（）
        fm[m,n]=(kk*ll[m,n])/(a*s[m,n])
        #求Q和D
        if m==0:     #北
          for l in np.arange(1,9):
            Q[k,m,l]=UV[k,2*m,l]*(q[k,m,l+1]-q[k,m,l-1])/2/d-UV[k,2*m+1,l]*(
                    q[k,m,l]-q[k,1,l])/d
            D[k,m,l]=fm[m,l]*(UV[k,2*m,l+1]-UV[k,2*m,l-1]+UV[k,2*m+1,l]-UV[k,2*m+5,l])/2/d
        if m==17:    #南
          for l in np.arange(1,9):
            Q[k,m,l]=UV[k,2*m,l]*(q[k,m,l+1]-q[k,m,l-1])/2/d-UV[k,2*m+1,l]*(
                    q[k,m-1,l]-q[k,m,l])/d  ###zheng fu hao
            D[k,m,l]=fm[m,l]*(UV[k,2*m,l+1]-UV[k,2*m,l-1]+UV[k,2*m-3,l]-UV[k,2*m+1,l])/2/d
        if n==0:     #东
          for l in np.arange(1,17):
            Q[k,l,n]=UV[k,2*l,n]*(q[k,l,1]-q[k,l,0])/d-UV[k,2*l+1,n]*(
                    q[k,l-1,n]-q[k,l+1,n])/2/d
            D[k,l,n]=fm[l,n]*(UV[k,2*l,n]-UV[k,2*l,n+2]+UV[k,2*l-1,n]-UV[k,2*l+3,n])/2/d
        if n==9:     #西
          for l in np.arange(1,17):
            Q[k,l,n]=UV[k,2*l,n]*(q[k,l,n]-q[k,l,n-1])/d-UV[k,2*l+1,n]*(
                    q[k,l-1,n]-q[k,l+1,n])/2/d
            D[k,l,n]=fm[l,n]*(UV[k,2*l,n-2]-UV[k,2*l,n]+UV[k,2*l-3,n]-UV[k,2*l+1,n])/2/d
  for i in np.arange(1,17):
      for j in np.arange(1,9):
        Q[k,i,j]=UV[k,2*i,j]*(q[k,i,j+1]-q[k,i,j-1])/2/d+UV[k,2*i+1,j]*(q[k,i,j+1]-q[k,i,j-1])/2/d
        D[k,i,j]=fm[i,j]*(UV[k,2*i,j+1]-UV[k,2*i,j-1]+UV[k,2*i-1,j]-UV[k,2*i+3,j])/2/d
  Q[k,0,0]=(Q[k,0,1]+Q[k,1,0])/2
  Q[k,0,9]=(Q[k,0,8]+Q[k,1,9])/2
  Q[k,17,0]=(Q[k,16,0]+Q[k,17,1])/2
  Q[k,17,9]=(Q[k,17,8]+Q[k,16,9])/2
  D[k,0,0]=(D[k,0,1]+D[k,1,0])/2
  D[k,0,9]=(D[k,0,8]+D[k,1,9])/2
  D[k,17,0]=(D[k,16,0]+D[k,17,1])/2
  D[k,17,9]=(D[k,17,8]+D[k,16,9])/2
#求水汽通量R
for k in np.arange(1,7):
    for i in np.arange(0,18):
        for j in np.arange(0,10):
            R[k,i,j]=(p[k]-p[k-1])*(Q[k,i,j]+q[k,i,j]*D[k,i,j])/g
    r=r+R
print('水汽通量','\n',r)









'''
T=np.empty((18,10,11))
for k in np.arange(0,11):
  for i in np.arange(0,18):
    for j in np.arange(0,10):
      T[i,j,k]=np.loadtxt('D:/ZD/DATA/678/T')
'''

'''
for file in files_t: #遍历文件夹
  position = path_t+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
  #print (position)
  with open(position, "r",encoding='utf-8') as f: #打开文件
    data0 = f.readlines() #读取文件
  for line in data0:
        line=line.strip('\n')
        data_split=line.split()
        temp=list(map(float,data_split))
        data_list.append(temp)
'''

'''
    #data=np.array(data)
    #txts.append(data)
#txts = ' '.join(txts)#转化为非数组类型
#import sys
#np.set_printoptions(threshold=sys.maxsize)  #全部显示
'''
#data1=np.array(data_list)
#print(data1)
#data=data1.reshape(11,18,10) #最终所用数组维度
#print(data)



