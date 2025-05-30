import numpy as np
from pandas import DataFrame
import pandas as pd

#import datetime
#from datetime import timedelta
#import string
#b=pd.read_table('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt',
#                sep=r'\s+',header=7)
b=pd.read_table('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs1.txt',
                sep=r'\s+',header=6)
#---sep参数，很重要---  \t，为逗号  r'\s+',为多个空格
c=b.iloc[5818::,:]
print('c','\n',c)

c['date'] =c['Date/Time'].str.split('/').str.get(0)
c['time'] =c['Date/Time'].str.split('/').str.get(1)

c['date']=pd.to_datetime(c['date'],format='%d-%m-%Y')
#c['Date/Time']=pd.to_datetime(c['Date/Time'],format='%d-%m-%Y %H:%M:%S')

y2001=c[c['date'].dt.year.isin([2001])]    #-----------------卡了好久

y2001count1=list(y2001['F012']).count(0.33)
y2001count2=list(y2001['F012']).count(0.67)
y2001count3=list(y2001['F012']).count(1.00)
y2001grouped=y2001['012hT01'].groupby(y2001['F012'])
y01_T36=y2001grouped.mean()

#y2001_36conut=y2001grouped.count(1.00)
#print('y2001conut1','\n',y2001count1)
#print('y2001conut2','\n',y2001count2)
#print('y2001conut3','\n',y2001count3)
#print('y01_T36','\n',y01_T36)

#y2002=c[c['date'].dt.year.isin([2002])]
#y2002grouped=y2002.groupby(y2001['STMID'])
#print('y2002grouped','\n',y2002grouped)
#y1=np.empty((16,7))
#y2=np.empty((16,7))
#y3=np.empty((16,7))

#ymeans=[]
#y=[]
z=['012hT01','024hT01','036hT01','048hT01','072hT01','096hT01','120hT01']
z1=[14,15,16,17,18,19,20]
n=0;s=[];tfe012=[]  #;ll=[];lll=[]
#def divisor(pp):
#    p=int(pp/3)
#    return p
error=[];number=[]
sd=[];r=[];f=[]

for i in np.arange(0,19):
    y=c[c['date'].dt.year.isin([2001+i])]   # 01-16年的error

    m=0
    yind=y['STMID'].value_counts()
    yin=list(yind.index)
    length=len(yin)

    #统计error 012
    s012=0;sc=0
    for f000 in y['000hT01']:
        if f000!=-9999.0: # and f000!=0.0:
            s012=s012+f000
            sc=sc+1
    tfe0=s012/sc
    tfe012.append(tfe0)


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
        yerror=yerrors/ynum            #每一年、每一个提前期的  error平均

        error.append(yerror)
        number.append(ynum)

        sdi=np.std(err,ddof=1)     #样本标准差
        sd.append(sdi)

        rx=0;ry=0
        for rr in np.arange(0,len(err)-1):
            rx=rx+(err[rr]-yerror)*(err[rr+1]-yerror)
            ry=ry+(err[rr]-yerror)**2
        ry=ry+(err[len(err)-1]-yerror)**2
        ri=rx/ry
        r.append(ri)               #自相关系数

        fi=np.sqrt((1+ri)/(1-ri))  #f
        f.append(fi)

number=np.array(number).reshape(19,7)
error=np.array(error).reshape(19,7)
sd=np.array(sd).reshape(19,7)
r=np.array(r).reshape(19,7)
f=np.array(f).reshape(19,7)

#ysum=y1+y2+y3                     # cases
#print('ysum','\n',ysum)
#print('ss','\n',np.array(s).reshape(16,7))
#print('y3','\n',y3)

#------------------------------------------
print('number','\n',number)
print('error','\n',error)
#print(np.array(err).mean())
print('sd','\n',sd)
print('r','\n',r)
print('f','\n',f)

#sem ： standard error of the mean--------------
sem=np.empty((19,7))
num_ind=np.empty((19,7))
w=np.empty((19,5))
for i in np.arange(0,19):
    for j in np.arange(0,7):
        sem[i,j]=sd[i,j]*f[i,j]/np.sqrt(number[i,j])  # sem
        num_ind[i,j]=int(number[i,j]/(f[i,j])**2)     # independent cases
    for k in np.arange(2,7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])                 # w
print('sem','\n',sem)
print('num_ind','\n',num_ind)
print('w','\n',w)
sem2=sem[:,2::]*2.18
print('sem2','\n',sem2)



# perceived error
tfe=[[41.53,73.84,103.77,135.56,206.55,259.21,344.74],
     [40.90,69.03,95.92,123.89,190.93,282.04,370.09],
     [35.99,61.31,87.19,116.87,138.78,165.55,204.95],
     [31.45,54.67,76.27,99.56,149.29,204.43,269.23],
     [32.14,56.41,80.21,100.54,148.31,221.20,301.22],
     [28.54,49.86,72.21,100.09,150.20,206.25,277.20],
     [30.02,47.31,66.63,84.64,124.90,153.40,227.01],
     [27.46,48.12,67.29,86.94,125.25,156.24,190.89],
     [31.76,44.63,60.48,72.75,111.32,192.66,289.58],
     [32.13,51.10,67.62,84.88,126.09,169.46,189.73],
     [27.55,42.80,56.90,70.41,110.49,170.16,253.68],
     [24.08,39.22,51.85,65.83,98.02,138.59,186.68],
     [28.02,49.54,71.81,103.17,136.07,133.49,170.09],
     [24.78,36.31,48.75,63.76,95.69,148.52,238.03],
     [24.52,39.15,53.17,72.76,109.67,165.50,238.34],
     [24.40,36.59,47.08,59.94,85.78,131.26,168.47]]
tfe=np.array(tfe)
#print('tfe','\n',tfe)
#print(err,num)

with open('D:/毕业设计/error.txt','w') as outfile:
    np.savetxt(outfile,error)
with open('D:/毕业设计/sem.txt','w') as outfile:
    np.savetxt(outfile,sem)

#Figure_1
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
fig=plt.figure()
ax1=fig.add_subplot(221)
label1=['36h','48h','72h','96h','120h']
cc=['orange','royalblue','forestgreen','deepskyblue','orangered']
#cc=['deepskyblue','orange','royalblue','forestgreen','orangered']
for x in np.arange(0,5):
    y=number[:,x+2]
    xx=np.arange(0,19)
    label=label1[x]
    plt.plot(xx,y,c=cc[x],lw=2,label=label)
ax1.legend( loc='lower center',bbox_to_anchor=(0.5,-0.4),
            ncol=5,frameon=False,fontsize=18)
#ax1.axis('off')
ax1.spines['top'].set_linewidth(0)
ax1.spines['right'].set_color('None')
ax1.spines['bottom'].set_linewidth(0)
ax1.spines['left'].set_color('None')
ax1.set_ylabel('台风个例数',rotation=90,fontsize=20,labelpad=23)
ax1.set_xticks(np.arange(0,19))
ax1.set_xticklabels(np.arange(2001,2020),rotation=90,fontsize=18)
ax1.set_yticks([0,100,200,300,400,500])
ax1.grid(axis='y',linestyle='-',lw=1)
ax1.tick_params(size=6,labelsize=18)
ax1.text(-4.2,500,'(a)',fontsize=18)


ax2=fig.add_subplot(222)
cc=['orange','royalblue','forestgreen','deepskyblue','orangered']
for x in np.arange(0,5):
    y=num_ind[:,x+2]
    xx=np.arange(0,19)
    label = label1[x]
    plt.plot(xx,y,c=cc[x],lw=2,label=label)
ax2.legend( loc='lower center',bbox_to_anchor=(0.5,-0.4),
            ncol=5,frameon=False,fontsize=18)
ax2.spines['top'].set_linewidth(0)
ax2.spines['right'].set_color('None')
ax2.spines['bottom'].set_linewidth(0)
ax2.spines['left'].set_color('None')
ax2.set_ylabel('独立的台风个例数',rotation=90,fontsize=20,labelpad=23)
ax2.set_xticks(np.arange(0,19))
ax2.set_xticklabels(np.arange(2001,2020),rotation=90,fontsize=18)
ax2.set_yticks(np.arange(0,100,10))
ax2.grid(axis='y',linestyle='-',lw=1)
ax2.tick_params(size=6,labelsize=18)
ax2.text(-3.9,90,'(b)',fontsize=18)


ax3=fig.add_subplot(223)
cc=['orange','royalblue','forestgreen','deepskyblue','orangered']
for x in np.arange(0,5):
    y=error[:,x+2]
    xx=np.arange(0,19)
    label = label1[x]
    plt.plot(xx,y,c=cc[x],lw=2,label=label)
ax3.legend( loc='lower center',bbox_to_anchor=(0.5,-0.4),
            ncol=5,frameon=False,fontsize=18)
ax3.spines['top'].set_linewidth(0)
ax3.spines['right'].set_color('None')
ax3.spines['bottom'].set_linewidth(0)
ax3.spines['left'].set_color('None')
ax3.set_ylabel('感知误差（海里）',rotation=90,fontsize=20,labelpad=23)
ax3.set_xticks(np.arange(0,19))
ax3.set_xticklabels(np.arange(2001,2020),rotation=90,fontsize=18)
ax3.set_yticks([0,100,200,300,400,500])
ax3.grid(axis='y',linestyle='-',lw=1)
ax3.tick_params(size=6,labelsize=18)
ax3.text(-4.2,500,'(c)',fontsize=18)


ax4=fig.add_subplot(224)
cc=['wheat','lightskyblue','bisque','palegreen',
    'silver','cornflowerblue','sandybrown','limegreen',
    'gray','deepskyblue','chocolate','lightcoral',
    'darkorange','orangered','dodgerblue','forestgreen','r','b','g']
for x in np.arange(0,19):
    y = error[x, 2::]
    xx=np.arange(0,5)
    label=x+2001
    plt.plot(xx,y,c=cc[x],lw=2,label=label)
ax4.legend( loc='lower center',bbox_to_anchor=(0.5,-0.5),
            ncol=6,frameon=False,fontsize=15)
ax4.spines['top'].set_linewidth(0)
ax4.spines['right'].set_color('None')
ax4.spines['bottom'].set_linewidth(0)
ax4.spines['left'].set_color('None')
ax4.set_ylabel('感知误差（海里）',rotation=90,fontsize=20,labelpad=23)
ax4.set_xticks(np.arange(0,5))
ax4.set_xticklabels(['36h','48h','72h','96h','120h'],fontsize=18)
ax4.set_yticks(np.arange(0,450,50))
ax4.grid(axis='y',linestyle='-',lw=1)
ax4.tick_params(size=6,labelsize=18)
ax4.text(-0.92,400,'(d)',fontsize=18)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 SimHei FangSong SimSun KaiTi
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.show()



fiterror=np.empty((16,5))
ysim=np.empty((16,5))
from scipy.optimize import curve_fit

x=np.array([36,48,72,96,120])

def nihe(x,a,b,c):
    a=45;xi = np.array([12, 12, 48, 48, 48])
    return ((a * a * np.exp(b * (x) /24) + c * c))/(w[i,:]**0.5)
    #if sim[:]<sem2[i,:]:
    #    return sim
    #if max(abs(sim[:]))<max(sem2[i,2::]):
    #   return sim
    #for k in np.arange(0,5):
    #    if sim[k]<sem2[i,2+k]:
    #        return sim
    #ysim=((a*a*np.exp(b*x)+c*c-error[i,j+2]**2)**2)/w[i,j]
    #return ysim
'''
    for i in np.arange(0,16):
      for j in np.arange(0,5):
        if j<2:
            ysim[i,j]=(a*a*np.exp(b*x[j])+c*c-error[i,j+2]**2)/w[i,j]
            fiterror[i,j]=abs(np.sqrt(a*a*np.exp(b*x[j])+c*c)-error[i,j+2])
        if j>=2:
            ysim[i,j]=(a*a*np.exp(b*x[j])+c*c-error[i,j+2]**2)/w[i,j]
            fiterror[i,j]=abs(np.sqrt(a*a*np.exp(b*x[j])+c*c)-error[i,j+2])
      for k in np.arange(0,5):
        if fiterror[i,k]<sem2[i,k+2]:
            if k<2:
                return a*a*np.exp(b*x[k])+c*c #-error[i,k+2]**2
            if k>=2:
                return a*a*np.exp(b*x[k])+c*c #-error[i,k+2]**2
'''




    #xi=[12,12,24,24,24]

    #return ((a*a*np.exp(b*(x)/24)+c*c))     #拟合函数
# -(x[i+1]-x[i])    x[i+1]-(x[i+1]-x[i])
#xx=[[60,70],[50,60],[50,60],[40,50],[40,50],[40,50],[40,50],[40,50],
#    [25,35],[40,50],[25,35],[25,35],[40,50],[20,30],[25,35],[25,35]]
#xy=[[20,25],[20,25],[20,25],[20,25],[15,25],[15,20],[15,20],[10,15],
#    [15,20],[10,15],[15,20],[15,20],[10,15],[10,15],[8,12],[5,10]]
xx=[[62,65],[52,56],[54,58],[43,46],[40,46],[40,46],[40,46],[43,46],
    [27,30],[42,46],[29,31],[28,30],[45,48],[22,25],[29,31],[29,31]]
xy=[[23,25],[20,22],[21,23],[21,23],[17,20],[16,18],[16,18],[14,15],
    [17,19],[14,15],[15,17],[15,17],[11,13],[12,14],[9,11],[7,9]]
xx=np.array(xx);xy=np.array(xy)
simulate=np.empty((16,5))
estimate=np.empty((16,8))

for i in np.arange(0,16):

    x=np.array([36,48,72,96,120])
    y=error[i,2::]                      #感知误差
    xi = np.array([12, 12, 48, 48, 48])

    popt,pcov=curve_fit(nihe,x,y**2/(w[i,:]*0.05),bounds=([xx[i,0],1.3,xy[i,0]],[xx[i,1],1.62,xy[i,1]]),
                        maxfev = 1000000)  #p0=(1,1,1)
    print(popt)
    x=np.array(x)
    x1 = np.array([0, 12, 24, 36, 48, 72, 96, 120])
    simulate[i] = np.sqrt((popt[0]*popt[0]*np.exp(popt[1]*(x-xi)/24)+popt[2]*popt[2]))
    #estimate[i] = nihe(x1, popt[0], popt[1], popt[2])**0.5
print('simulate',simulate)
#print('estimate',estimate)



coe_last=np.zeros((3))
coe_start=np.zeros((3))
coe_int=np.zeros((3))

coe_start[0]=10;coe_start[1]=0.5;coe_start[2]=5           #初值
coe_int[0]=0.1;coe_int[1]=0.01;coe_int[2]=0.05         #间隔

sim=np.empty((16,5))
sim_err=np.empty((16,5))
xbest=np.zeros((16,3))
simmin=1e20

'''
for i in np.arange(0,750):
    coe_last[0]=coe_start[0]+coe_int[0]*i
    for j in np.arange(0,150):
        coe_last[1]=coe_start[1]+coe_int[1]*j
        for k in np.arange(0,500):
            coe_last[2]=coe_start[2]+coe_int[2]*k
            for m in np.arange(0,16):
                q=0;fmax=0
                for n in np.arange(0,5):
                    #sim[m,n]=
                    #x=np.array([36,48,72,96,120])
                    if n<2:
                        sim[m,n]=((coe_last[0]**2*np.exp(coe_last[1]*(n+3)*12/24)
                                    +coe_last[2]**2-error[m,n+2]**2)**2)/w[m,n]
                        sim_err[m,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(n+3)*12/24)
                                         -error[m,n+2])
                    if n>=2:
                        sim[m,n]=((coe_last[0]**2*np.exp(coe_last[1]*(n+1)*24/24)
                                    +coe_last[2]**2-error[m,n+2]**2)**2)/w[m,n]
                        sim_err[m,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(n+1)*24/24)
                                         -error[m,n+2])
                    if sim_err[m,n]<sem2[m,n]:
                        q=q+1
                    if sim[m,n]>fmax:
                        fmax=sim[m,n]
                if simmin>fmax and q==5:
                    simmin=fmax
                    xbest[m,0]=coe_last[0]
                    xbest[m,1]=coe_last[1]
                    xbest[m,2]=coe_last[2]
                    #print(xbest)
                    print('hello,world!',m)
print(xbest)
print(coe_last)
ysimulate=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate[i,j]=(xbest[i,0]**2*np.exp(xbest[i,1]*(j+3)*12/24)
                                   +xbest[i,2]**2)**0.5
        if j>=2:
            ysimulate[i,j]=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(j+1)*24/24)
                                   +xbest[i,2]**2)
    #ysimulate=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(x)/24)+xbest[i,2]**2)
print(ysimulate)
'''

x=np.array([36,48,72,96,120])
#y=error[0,2::]
coe_last=np.array([60,1,25])
#print(coe_last[0])
xi = np.array([12, 12, 24, 24, 24])
y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
step=0.1
amp=0.1
simulate1=np.empty((16,5))
'''
for i in np.arange(0,16):
  for j in np.arange(0,10000):
    wg=0.01
    coe_add=coe_last+wg
    y_add=(coe_add[0]**2*np.exp(coe_add[1]*(x)/24)+coe_add[2]**2)**0.5
    error_add=np.std(error[0,2::]-y_add)

    coe_sub=coe_last-wg
    y_sub=(coe_sub[0]**2*np.exp(coe_sub[1]*(x)/24)+coe_sub[2]**2)**0.5
    error_sub=np.std(error[0,2::]-y_sub)

    error_e=(error_add-error_sub)*step*wg
    coe_last=coe_last-error_e

    simulate1[i,:]=abs(coe_last[0]*np.exp(coe_last[1]*(x)/24/2)-error[0,2::])
    for k in np.arange(0,5):
        q=0
        if simulate1[i,n]<sem2[i,n]:
            q=q+1
    if q==5:
            print(simulate1[i,:],'finish',coe_last)


'''
#print(error_e)
#print(coe_last)
#print(np.sqrt(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2))
print('hello,world!')


#independent cases  计数---第二层-----------------------------------
'''
    for j in ['F012', 'F024', 'F036', 'F048', 'F072', 'F096', 'F120']:
        y1[i, m] = list(y[j]).count(0.33)
        y2[i, m] = list(y[j]).count(0.67)
        y3[i, m] = list(y[j]).count(1.00)

        ygrouped = y[z[m]].groupby(y[j])
        ymean = ygrouped.mean()
        ymeans.append(2001 + i)
        ymeans.append(y1[i, m])
        ymeans.append(y2[i, m])
        ymeans.append(y3[i, m])
        ymeans.append(ymean)

        m = m + 1
    n = n + 1
ss = np.array(s).reshape(16, 7)  # independent cases
'''

# indepentent cases  算法1----第三层----------------------------------------
'''
    r=0;q=0;t=0
    for f in y[j]:
        if f==1.00:
            t=t+1
            if q>3:
                r=int(q/4)
                t=t+r
            q=0
        if f==0.33 or 0.67:
            q=q+1
        if f==0.00:
            if q>3:
                r=int(q/4)
                t = t + r
            q=0

    s.append(t)

# independent cases  算法2----第三层-------------------------------------

    t = 0;w=0;l=0
    for k in np.arange(0, length):
        #    yi=y['STMID'][yin[k]]
        yi = y[y['STMID'].isin([yin[k]])]

        p=0;q=0;r=0;u=0

        for f in yi[j]:
            u = u + 1
            if f!=0:
                p=p+1
            else:
                if p >= 3:
                    q = int(p / 4)
                    r = r + q
                p = 0
                v=yi._get_value(u-1,z1[m],takeable=True)#.get_value
                if v!=0 and v!=-9999.0:
                    w = w + v
                    l=l+1
        t = t + r

    #s.append(t)
    #ll.append(w/l)
    #lll.append(l)
'''

# 拟合 1 --------------------------------------------------------
'''
x=np.array([36,48,72,96,120])
x2=np.array([0,12,24,36,48,72,96,120])
y=tfe[0,2::]
popt,pcov=curve_fit(nihe,x,y*y,bounds=([60,1.2,20],[70,1.7,25]),maxfev=1000000)#
print(popt)
y1=nihe(x,popt[0],popt[1],popt[2])**0.5
y2=nihe(x2,popt[0],popt[1],popt[2])**0.5
print(y1)
print(y2)
'''
#定义函数
'''
def FE(x,m,n):  #forecast error
  y=c[c['date'].dt.year.isin([x])]    #-----------------卡了好久
  y1=list(y['m']).count(1.00)
  y3=list(y['m']).count(0.33)
  y2001grouped=y['n'].groupby(y['m'])  #m--F036,n--036hT01
  yy=y2001grouped.mean()
'''

#有用
'''
td=timedelta(days=7)
tt=datetime.datetime.now()+td
print(tt)
'''

#c['Date/Time']=pd.to_datetime(c['Date/Time'])
#print(c['Date/Time'])
#for i in np.arange(1,4831):
#  c[i,1].strftime('%Y-%m-%d %H:%M:%S')
#print(c['Date/Time'],'xxxxx')

#y2001=c['Date/Time'].dt.year.isin([2001])


#groupby2001=c['Lat'].groupby(c['STMID']).mean
#print('groupby','\n',groupby2001)

#if c['Date/Time'].dt.year==2001:
#    y2001=y2001.apply()



#时间格式处理-尝试--------------------------------------------------
'''
#pd.to_datetime(c.loc[:,['Date/Time']])
#c.loc[:,['Date/Time']].strftime('%j')
#c['Date/Time']=datetime.strptime(c['Date/Time'],'%Y-%m-%d %H-%M-%S')
'''

'''
print('xxx')
with open('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt') as f:
    lines_after= f.readlines()[5542:]
forecast_error=np.array(lines_after)
#print(forecast_error)
print(forecast_error.shape)

#如下参照 诊断课-水汽
with open('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt'
        , "r", encoding='utf-8') as f:  # 打开文件
    data0 = f.readlines()[5542:]  # 读取文件
data_list = []
for line in data0:
    line = line.strip('\n')
    data_split = line.split()
    temp = list(map(float, data_split))
    data_list.append(temp)
data1 = np.array(data_list)
data = data1.reshape(5115,54)  # 最终所用数组维度
print(data)

import itertools
#with open('D:/毕业设计/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt') as f:
#    for line in itertools.islice(f, 17, None):

'''
