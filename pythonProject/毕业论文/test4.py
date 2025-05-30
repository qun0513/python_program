import numpy as np
error=np.loadtxt('D:/毕业设计/error.txt')
sem=np.loadtxt('D:/毕业设计/sem.txt')
sem2=sem*2.18
w=np.empty((16,5))
for i in np.arange(0,16):
    for k in np.arange(2, 7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])

#coe_last=np.zeros((3))
#coe_start=np.zeros((3))
#coe_int=np.zeros((3))

#coe_start[0]=15;coe_start[1]=0.5;coe_start[2]=0           #初值
#coe_int[0]=0.1;coe_int[1]=0.01;coe_int[2]=0.001         #间隔

sim=np.empty((16,5))
sim_err=np.empty((16,5))
xbest=np.zeros((4))
xbest1=np.zeros((16,4))
xbest2=np.zeros((4))
#simmin=1e20



x=np.array([36,48,72,96,120])
#coe_last=np.array([60,1,25])
xi = np.array([12, 12, 24, 24, 24])
#y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
step=0.8
amp=0.1

noptimum=np.zeros((16))
coe_last=np.empty((4))
from numpy import random
#模型试验------------------------------------------------------------------
f4=np.empty((16))
for i in np.arange(0,16):
  coe_last[0]=57.21297616;coe_last[1]=0.95551215;coe_last[2]=0.7231428;coe_last[3]=0.35882945  #固定
  #    x0                   b                    a               r
  simmin = 1e20

  #y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
  for j in np.arange(0,2000):  #250000
    #wg=np.array([0.001,0.00009,0.00012,0.00011])
    wg=5*np.array([random.uniform(0.001,0.0015),random.uniform(0.0001,0.00015),
          random.uniform(0.0001,0.00015),random.uniform(0.0001,0.00015)])

    coe_add=coe_last+wg
    y_add=(coe_add[0]**2*coe_add[1]**(2*i)*np.exp(coe_add[2]*(x)/24)
           +coe_add[3]*coe_add[0]**2*coe_add[1]**(2*i))**0.5
    error_add=np.std(error[i,2::]-y_add)

    coe_sub=coe_last-wg
    y_sub=(coe_sub[0]**2*coe_sub[1]**(2*i)*np.exp(coe_sub[2]*(x)/24)
           +coe_sub[3]*coe_sub[0]**2*coe_sub[1]**(2*i))**0.5
    error_sub=np.std(error[i,2::]-y_sub)

    error_e=(error_add-error_sub)*step*wg
    coe_last=coe_last-error_e

    q=0;fmax=0
    for n in np.arange(0,5):
        if n < 2:
            sim[i,n]=((coe_last[0]**2*coe_last[1]**(i*2)*
                       np.exp(coe_last[2]*(n+3)*12/24)
                       +coe_last[3]*coe_last[0]**2*coe_last[1]**(i*2)
                       -error[i,n+2]**2)**2)/w[i,n]

            sim_err[i,n]=abs(coe_last[0]*coe_last[1]**(i)*
                             np.exp(0.5*coe_last[2]*(n+3)*12/24)
                             -error[i,n+2])

        if n >= 2:
            sim[i,n]=((coe_last[0]**2*coe_last[1]**(i*2)*
                       np.exp(coe_last[2]*(n+1)*24/24)
                       +coe_last[3]*coe_last[0]**2*coe_last[1]**(i*2)
                       -error[i,n+2]**2)**2)/w[i,n]

            sim_err[i,n]=abs(coe_last[0]*coe_last[1]**(i)*
                             np.exp(0.5*coe_last[2]*(n+1)*24/24)
                             -error[i,n+2])

        if sim_err[i, n] < sem2[i, n]:
            q = q + 1
        if sim[i, n] > fmax:
            fmax = sim[i, n]
    if simmin > fmax and q == 5:
            simmin=fmax
            xbest[0]=coe_last[0]
            xbest[1]=coe_last[1]
            xbest[2]=coe_last[2]
            xbest[3]=coe_last[3]
            #print(xbest)
            f4[i]=simmin
            print('hello,world!',i)
            #noptimum[i]=noptimum[i]+1
    simmin=1e20
    if simmin > fmax and q == 5:
            xbest1[i,0] = coe_last[0]
            xbest1[i,1] = coe_last[1]
            xbest1[i,2] = coe_last[2]
            xbest1[i,3] = coe_last[3]
            #print(xbest)
            print('Hello,world!',i)
            #coe1_max = coe_last[1]
            #if coe_last[1]>1:
            #    xbest2[i, 0] = coe_last[0]
            #    xbest2[i, 1] = coe_last[1]
            #    xbest2[i, 2] = coe_last[2]
  #print(i)
  print(i)
print('f4','\n',f4)
print('xbest','\n',xbest)
print('xbest1','\n',xbest1)
#print('xbest2','\n',xbest2)
#print(noptimum)

ysimulate=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate[i,j]=np.sqrt(xbest[0]**2*xbest[1]**(2*i)*
                       np.exp(xbest[2]*(j+3)*12/24)
                       +xbest[3]*xbest[0]**2*xbest[1]**(2*i))
        if j>=2:
            ysimulate[i,j]=np.sqrt(xbest[0]**2*xbest[1]**(2*i)*
                       np.exp(xbest[2]*(j+1)*24/24)
                       +xbest[3]*xbest[0]**2*xbest[1]**(2*i))
    #ysimulate=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(x)/24)+xbest[i,2]**2)
print('ysimulate','\n',ysimulate)


ysimulate1=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate1[i,j]=np.sqrt(xbest1[i,0]**2*xbest1[i,1]**(2*i)*
                       np.exp(xbest1[i,2]*(j+3)*12/24)
                       +xbest1[i,3]*xbest1[i,0]**2*xbest1[i,1]**(2*i))
        if j>=2:
            ysimulate1[i,j]=np.sqrt(xbest1[i,0]**2*xbest1[i,1]**(2*i)*
                       np.exp(xbest1[i,2]*(j+1)*24/24)
                       +xbest1[i,3]*xbest1[i,0]**2*xbest1[i,1]**(2*i))
    #ysimulate=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(x)/24)+xbest[i,2]**2)
print('ysimulate1','\n',ysimulate1)

'''
for i in np.arange(0,16):
    if f4[i]<1:
        f4[i]=1e200
    p4=min(f4)
for j in np.arange(0,16):
    if f4[j]==p4:
        p4index=j
parameter4=np.empty((4))
#print(xbest1[p4index,:])
parameter4=xbest1[p4index,:]
#print(parameter4)
coe_last=parameter4
print(coe_last)
'''

p4max=np.empty((16))
sim1=np.empty((16,5))
coe_last1=np.empty((4))
p4=[]
for k in np.arange(0,16):
    if xbest1[k, 0] != 0:
        coe_last1=xbest1[k,:]
        for i in np.arange(0,16):
          for n in np.arange(0,5):
            if n < 2:
                sim1[i,n]=((coe_last1[0]**2*coe_last1[1]**(i*2)*
                       np.exp(coe_last1[2]*(n+3)*12/24)
                       +coe_last1[3]*coe_last1[0]**2*coe_last1[1]**(i*2)
                       -error[i,n+2]**2)**2)/w[i,n]



            if n >= 2:
                sim1[i,n]=((coe_last1[0]**2*coe_last1[1]**(i*2)*
                       np.exp(coe_last1[2]*(n+1)*24/24)
                       +coe_last1[3]*coe_last1[0]**2*coe_last1[1]**(i*2)
                       -error[i,n+2]**2)**2)/w[i,n]
          p4max[i]=max(sim1[i,:])
        p4min=min(p4max)
        p4.append(p4min)

print(p4)
print(min(p4))

'''
for i in np.arange(0,16):
  coe_last[0]=64;coe_last[1]=-1.0;coe_last[2]=0.8;coe_last[3]=0.4  #固定
  #    x0             b                a               r
  simmin = 1e20

  #y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
  for j in np.arange(0,100000):  #250000
    wg=np.array([0.001,0.00009,0.00012,0.00011])

    coe_add=coe_last+wg
    y_add=(coe_add[0]**2*np.exp(coe_add[1]*(i))*np.exp(coe_add[2]*(x)/24)
           +coe_add[3]*coe_add[0]**2*np.exp(coe_add[1]*(i)))**0.5
    error_add=np.std(error[i,2::]-y_add)

    coe_sub=coe_last-wg
    y_sub=(coe_sub[0]**2*np.exp(coe_sub[1]*(i))*np.exp(coe_sub[2]*(x)/24)
           +coe_sub[3]*coe_sub[0]**2*np.exp(coe_sub[1]*(i)))**0.5
    error_sub=np.std(error[i,2::]-y_sub)

    error_e=(error_add-error_sub)*step*wg
    coe_last=coe_last-error_e

    q=0;fmax=0
    for n in np.arange(0,5):
        if n < 2:
            sim[i,n]=((coe_last[0]**2*np.exp(coe_last[1]*(i))*
                       np.exp(coe_last[2]*(n+3)*12/24)
                       +coe_last[3]*coe_last[0]**2*np.exp(coe_last[1]*(i))
                       -error[i,n+2]**2)**2)/w[i,n]

            sim_err[i,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(i))*
                             np.exp(0.5*coe_last[2]*(n+3)*12/24)
                             -error[i,n+2])

        if n >= 2:
            sim[i,n]=((coe_last[0]**2*np.exp(coe_last[1]*(i))*
                       np.exp(coe_last[2]*(n+1)*24/24)
                       +coe_last[3]*coe_last[0]**2*np.exp(coe_last[1]*(i))
                       -error[i,n+2]**2)**2)/w[i,n]

            sim_err[i,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(i))*
                             np.exp(0.5*coe_last[2]*(n+1)*24/24)
                             -error[i,n+2])

        if sim_err[i, n] < sem2[i, n]:
            q = q + 1
        if sim[i, n] > fmax:
            fmax = sim[i, n]
        if simmin > fmax and q == 5:
            simmin=fmax
            xbest[i,0]=coe_last[0]
            xbest[i,1]=coe_last[1]
            xbest[i,2]=coe_last[2]
            #print(xbest)
            print('hello,world!',i)
            noptimum[i]=noptimum[i]+1

        if q==5:
            xbest1[i,0]=coe_last[0]
            xbest1[i,1]=coe_last[1]
            xbest1[i,2]=coe_last[2]
            #print(xbest)
            print('Hello,world!',i)
            #coe1_max = coe_last[1]
            #if coe_last[1]>1:
            #    xbest2[i, 0] = coe_last[0]
            #    xbest2[i, 1] = coe_last[1]
            #    xbest2[i, 2] = coe_last[2]
  #print(i)
  print(i)
print('xbest','\n',xbest)
print('xbest1','\n',xbest1)
#print('xbest2','\n',xbest2)
print(noptimum)

ysimulate=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate[i,j]=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(i))*
                       np.exp(xbest[i,2]*(j+3)*12/24)
                       +xbest[i,3]*xbest[i,0]**2*np.exp(xbest[i,1]*(i)))
        if j>=2:
            ysimulate[i,j]=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(i))*
                       np.exp(xbest[i,2]*(j+1)*24/24)
                       +xbest[i,3]*xbest[i,0]**2*np.exp(xbest[i,1]*(i)))
    #ysimulate=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(x)/24)+xbest[i,2]**2)
print('ysimulate','\n',ysimulate)

ysimulate1=np.empty((16,5))
for i in np.arange(0,16):
    for j in np.arange(0,5):
        if j<2:
            ysimulate1[i,j]=np.sqrt(xbest1[i,0]**2*np.exp(xbest1[i,1]*(i))*
                       np.exp(xbest1[i,2]*(j+3)*12/24)
                       +xbest1[i,3]*xbest1[i,0]**2*np.exp(xbest1[i,1]*(i)))
        if j>=2:
            ysimulate1[i,j]=np.sqrt(xbest1[i,0]**2*np.exp(xbest1[i,1]*(i))*
                       np.exp(xbest1[i,2]*(j+1)*24/24)
                       +xbest1[i,3]*xbest1[i,0]**2*np.exp(xbest1[i,1]*(i)))
    #ysimulate=np.sqrt(xbest[i,0]**2*np.exp(xbest[i,1]*(x)/24)+xbest[i,2]**2)
print('ysimulate1','\n',ysimulate1)
'''