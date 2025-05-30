import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

b=pd.read_table('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs1.txt',
                sep=r'\s+',header=6)

b['date'] =b['Date/Time'].str.split('/').str.get(0)
b['time'] =b['Date/Time'].str.split('/').str.get(1)
b['date']=pd.to_datetime(b['date'],format='%d-%m-%Y')

z=['012hT01','024hT01','036hT01','048hT01','072hT01','096hT01','120hT01']
error=[];number=[]
sd=[];r=[];f=[]

for i in np.arange(0,50):
    y=b[b['date'].dt.year.isin([1970+i])]   # 01-16年的error

    m=0
    yind=y['STMID'].value_counts()
    yin=list(yind.index)
    length=len(yin)



    for k in np.arange(0,7):
        ynew=list(y[z[k]])
        ylen=len(ynew)
        yerrors=0;ynum=0

        err = []
        for kk in np.arange(0,ylen):
            if ynew[kk]!=-9999.0:
                yerrors=yerrors+ynew[kk]
                ynum=ynum+1            #每一年、每一个提前期的  个数
                err.append(ynew[kk])   #每一年、每一个提前期的  每个error
        #if ynum!=0:
        yerror=yerrors/(ynum+0.00001)            #每一年、每一个提前期的  error平均

        error.append(yerror)
        number.append(ynum)

number=np.array(number).reshape(50,7)
error=np.array(error).reshape(50,7)
fig,ax=plt.subplots()
x=np.arange(0,50)
plt.plot(x,error[:,1],label='24h预报误差')
plt.plot(x,error[:,4],label='48h预报误差')

plt.legend()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong SimSun KaiTi
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.show()