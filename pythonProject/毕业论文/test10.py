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
#模型试验--20----------------------------------------------------------------
f4=np.empty((16))
for i in np.arange(0,16):
  coe_last[0] =56.2473877;coe_last[1] =0.96296054;coe_last[2]=0.7244406;coe_last[3]=0.3450621  #固定
  #    x0                   b                    a               r
  simmin = 1e20

  #y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
  for j in np.arange(0,2000):  #250000
    #coe_last[2] = 0.783356
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
        #coe_last[0] =60.2473877;coe_last[1] =0.95596054;coe_last[2]=0.7244406;coe_last[3]=0.35450621
        #coe_last[2] = 0.783356
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

parameter20=np.array([56.18,0.957,0.358,0.73623,0.7600,0.5414,
                       0.7048,0.7256,0.79344,0.7008,0.7080,0.7255,0.6799,
                       0.8444,0.7107,0.6096,0.8735,0.7249,0.7166])
print(np.mean(parameter20[3:18])*2)
'''
parameter20=np.array([52.59914,0.96059,0.38739,
                      
                      
                      0.7184,
                      0.7387,
                      
                      0.7074,
                      0.7149,
                      0.7327,
                      
                      
                      
                      0.7158,
                      
                      
                      0.7292,
                      0.7208])
print()
'''