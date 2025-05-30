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
xbest=np.zeros((16,3))
xbest1=np.zeros((16,3))
xbest2=np.zeros((16,3))
#simmin=1e20

#xx=[[62,65],[52,56],[54,58],[43,46],[40,46],[40,46],[40,46],[43,46],
#    [27,30],[42,46],[29,31],[28,30],[45,48],[22,25],[29,31],[29,31]]
#xy=[[23,25],[20,22],[21,23],[21,23],[17,20],[16,18],[16,18],[14,15],
#    [17,19],[14,15],[15,17],[15,17],[11,13],[12,14],[9,11],[7,9]]

x=np.array([36,48,72,96,120])
#y=error[0,2::]
coe_last=np.array([60,1,25])
#print(coe_last[0])
xi = np.array([12, 12, 24, 24, 24])
y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
step=0.8
amp=0.1

noptimum=np.zeros((16))

#模型试验--35----------------------------------------------------------------
from numpy import random
for i in np.arange(0,16):
  coe_last[0]=64-3*i;coe_last[1]=0.85;coe_last[2]=0.5
  simmin = 1e20

  #y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2)**0.5
  for j in np.arange(0,200):  #250000
    #coe_last[2] = 0.7
    #wg=np.array([0.001,0.0001,0.0001])
    wg = np.array([random.uniform(0.001, 0.0015), random.uniform(0.0001, 0.00015),
                   random.uniform(0.0001, 0.00015)])#, random.uniform(0.0001, 0.00015)])
    coe_add=coe_last+wg
    y_add=(coe_add[0]**2*np.exp(coe_add[1]*(x)/24)+coe_add[2]*coe_add[0]**2)**0.5
    error_add=np.std(error[i,2::]-y_add)

    coe_sub=coe_last-wg
    y_sub=(coe_sub[0]**2*np.exp(coe_sub[1]*(x)/24)+coe_sub[2]*coe_sub[0]**2)**0.5
    error_sub=np.std(error[i,2::]-y_sub)

    error_e=(error_add-error_sub)*step*wg
    coe_last=coe_last-error_e

    q=0;fmax=0
    for n in np.arange(0,5):
        #coe_last[2] = 0.7
        if n < 2:
            sim[i,n]=((coe_last[0]**2*np.exp(coe_last[1]*(n+3)*12/24)
                       +coe_last[2]*coe_last[0]**2-error[i,n+2]**2)**2)/w[i,n]
            sim_err[i,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(n+3)*12/24)
                                         -error[i,n+2])
        if n >= 2:
            sim[i,n]=((coe_last[0]**2*np.exp(coe_last[1]*(n+1)*24/24)
                        +coe_last[2]*coe_last[0]**2-error[i,n+2]**2)**2)/w[i,n]
            sim_err[i,n]=abs(coe_last[0]*np.exp(0.5*coe_last[1]*(n+1)*24/24)
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
    '''
    if q==5:
            xbest1[i,0]=coe_last[0]
            xbest1[i,1]=coe_last[1]
            xbest1[i,2]=coe_last[2]
            #print(xbest)
            print('Hello,world!',i)
            coe1_max = coe_last[1]
            if coe_last[1]>1:
                xbest2[i, 0] = coe_last[0]
                xbest2[i, 1] = coe_last[1]
                xbest2[i, 2] = coe_last[2]
    '''
  #print(i)
  print(i)
print('xbest','\n',xbest)
#print('xbest1','\n',xbest1)
#print('xbest2','\n',xbest2)
#print(noptimum)

parameter35=np.array([[62.0435,0.7362,0.3655],
                      [56.6569,0.7657,0.3657],
                      [60.4328,0.5414,0.2433],
                      [47.1060,0.7094,0.3106],
                      [47.4794,0.7479,0.3479],
                      [41.9213,0.7921,0.3922],
                      [40.8579,0.6858,0.2858],
                      [38.0737,0.7074,0.3074],
                      [36.1231,0.8124,0.4123],
                      [40.8013,0.6799,0.3802],
                      [30.4675,0.8468,0.4468],
                      [28.6000,0.7600,0.4600],
                      [48.1213,0.6096,0.2122],
                      [28.5825,0.8579,0.3584],
                      [31.0833,0.8093,0.5080],
                      [27.3489,0.7353,0.4348]]) # r=0.36444375
print('35_mean',np.mean(parameter35[:,2]),np.mean(parameter35[:,1])*2)

parameter35[:,2]=np.mean(parameter35[:,2])
print(parameter35)