#数据读取---------------------------------------------------------------
import os
import xarray as xr
import numpy as np
import math
path = 'D:/dyhl/data_60_89_CTL'         #控制实验
#path = 'D:/dyhl/data_60_89_Wind'        #风应力实验
#path = 'D:/dyhl/data_60_89_Water'       #淡水通量实验
ff=[]
for filename in os.listdir(path):
    full_path = os.path.join(path,filename)
    f = xr.open_dataset(full_path,decode_times=False)
    ff.append(f.ts[0,0,:,:])     #变量
fff=np.array(ff).reshape(360,115,182)
print(f.ts[0,0,:,:])
'''
如果读取nc文件时程序报错“unable to decode time units..."，即无法解析时间单位时，原因可能在于该nc文件是模式输出文件，时间通常是从0开始的，导致xarray模块无法解析，解决方案如下：
f=xr.open_dataset("xxx.nc",decode_times=False) #添加decode_times=False
'''
lat=f.ts.lat.data
lat[0]=89.99       #考虑到python中 cos（90） 误差较大，因此用89.99代替
pi=3.14;a=6357000
print(math.cos(89.99*pi/180),math.cos(90*pi/180))
dx=np.empty(115)     #x方向网格距
dy=np.empty(115)     #y方向网格距
s=np.empty(115)      #不同纬圈单个格点面积

for j in np.arange(0,114):
    dx[j]=2*pi*(a*math.cos((lat[j])*pi/180))*(2/360)
    dy[j]=2*pi*a*(abs(lat[j]-lat[j+1])/360)
    s[j]=dx[j]*dy[j]
s[114]=2*pi*(a*math.cos((78)*pi/180))*(2/360)*2*pi*a*(2/360)  #第115个纬圈上，单个格点的面积

#计算海表温度------------------------------------------------------
ss=0
for j in np.arange(0,115):
    x=0
    for k in np.arange(0,182):
        if not np.isnan(fff[0, j, k]):
            x = x + 1    #各纬圈海洋格点个数
    ss=ss+s[j]*x         #海洋总表面积

sst=np.empty(360)
for i in np.arange(0,360):
    sst0=0
    for j in np.arange(0,115):
        for k in np.arange(0,182):
            if not np.isnan(fff[i, j, k]):
                sst1=fff[i,j,k]           #有海表温度的格点
                sst0=sst0+sst1*(s[j]/ss)  #面积加权
    sst[i]=sst0       #面积加权后的海表温度
print(sst)

#绘图---------------------------------------------------------------
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签 SimHei FangSong(可能更普适)
plt.rcParams['axes.unicode_minus']=False     #用来正常显示负号

fig,ax=plt.subplots()
ax.plot(np.arange(0,360),sst,c='r',lw=2,label='Temperature')
ax.set_title('1960-1989_global_average_SST(ctl)',fontsize=32,pad=30)
ax.set_xlabel('year',fontsize=30,labelpad=16)
ax.set_ylabel('Temperature',fontsize=30,rotation=90,labelpad=16)
ax.set_xticks(np.arange(0,361,60))
ax.set_xticklabels(np.arange(1960,1991,5))
ax.set_yticks(np.arange(17.7,19.0,0.1))

ax.tick_params(size=6,labelsize=20,pad=12)

#plt.legend(fontsize=16)
plt.show()


# 合并文件-----------------------------------------------------------------
''''
# merge all files into a new file.

import xarray as xr
import os

def merge(path):
    file=os.listdir(path)
    ds=xr.open_dataset(path+file[0],decode_times=False)

    for i in range(len(file)-1):
        dsi=xr.open_dataset(path+file[i+1],decode_times=False)
        ds=xr.concat([ds,dsi],dim='time')
        dsi.close()
        print(file[i+1][-8:-3]+' merged')

    return ds

path='E:/data_60_90_CTL/data_60_90_CTL/'

ds=merge(path)
ds.to_netcdf(path+'D:/Merged.nc')
print('success')
'''

'''
import os
path="E:/data_60_90_CTL/data_60_90_CTL/"
files=os.listdir(path)
for file in files:
    f=
    #f=open(path+file,"r")
'''

#ds=xr.open_dataset(path)
#print(f.ts[:,0,:,:].mean(dim='time'))
#print(f.ts)
#print(fff.mean(axis=0))
#sst=fff.mean(axis=1)
#print(sst.shape)


# 数据的初步处理 ---------------------------------------------------
'''
sf=np.empty(115)
#print('dx',dx,'dy',dy)
for k in np.arange(0,115):
    sf[k]=s[k]/sum(s)
#print(s)

sst1=np.empty((31,115,182))
#sst2=np.empty((31,115))
sst3=np.empty((31,115))
count=np.empty(115)
sst0=np.empty(31)
for i in np.arange(0,31):
    sss = 0
    sns=0
    for j in np.arange(0,115):
        x=0
        for k in np.arange(0,182):
            if not np.isnan(sst[i,j,k]):
                x=x+1
        sst2=np.mean(sst[i,j,:])
        sn=x*s[j]*sst2
        sns = sns + sn
        ss = x * s[j]
        sss=sss+ss
        #print(sn)

    #print('x',sns,sss)

    sst0[i]=sns/sss
#sst0=sum(sst3[:,:])/sum(count[:]*s[:])
        #sst1[i,j,:]=sst[i,j,:]*sf[j]
        #sst2[i,j]=np.mean(np.nan_to_num(sst1[i,j,:]))
    #sst2[i]=sum(list(sst1[i,:,:].mean(axis=1,np.nan_to_num(a))))
    #print(sst1)
    #sst3[i]=sum(sst2[i,:])

print(sst0)
'''

'''
    if i.data>30:
        s[j]=dx[j]*(dy*2)
        j=j+1
    elif -30<i.data<30:
        s[j]=dx[j]*dy
        j=j+1
    else:
        s[j]=dx[j]*(dy*2)
        j=j+1
'''