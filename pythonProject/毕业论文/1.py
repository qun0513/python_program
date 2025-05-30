import numpy as np
a=[[1,2,3],
   [4,5,6]]
b=[1,2,3]
c=b.remove(1)
x=0
#print(a[1,x+1::])
#print(np.array(a).shape)
#print(c)

xx=np.array([[1,2,3,4,5,6],[6,5,5,3,2,1]])
#print(sum(xx[0,:]),sum(xx[1,:]))

d=[1,2,3,4,5]
e=[1,2,3,4,5]
wg=np.random.rand(1)
#print(wg)
a1=np.ones((3))
#print(a1)

coe_last=np.array([60,1,25])
wg=np.array([0.01,0.0001,0.005])
coe_add=coe_last+wg
#print(coe_add)
y1=np.array([1,2,3,4,5])
y2=np.array([2.5,3,4,4.5,6])
#print(np.std(y2 - y1))

error=np.loadtxt('D:/毕业设计/error.txt')
sem=np.loadtxt('D:/毕业设计/sem.txt')

sem2=sem*2.18
w=np.empty((16,5))
for i in np.arange(0,16):
    for k in np.arange(2, 7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])
print(w[0,:])
sim_err=np.empty((5))
sim=np.empty((5))
coe_last=np.array([27.21782488,0.74785499,7.1327496])
for i in np.arange(0,5):
   if i < 2:
      sim[i] = ((coe_last[0] ** 2 * np.exp(coe_last[1] * (i + 3) * 12 / 24)
                    + coe_last[2] ** 2 - error[15, i + 2] ** 2) ** 2) / w[15, i]

      sim_err[i] = abs(coe_last[0] * np.exp(0.5 * coe_last[1] * (i + 3) * 12 / 24)
                          - error[15, i + 2])
   if i >= 2:
      sim[i] = ((coe_last[0] ** 2 * np.exp(coe_last[1] * (i + 1) * 24 / 24)
                 + coe_last[2] ** 2 - error[15, i + 2] ** 2) ** 2) / w[15, i]

      sim_err[i] = abs(coe_last[0] * np.exp(0.5 * coe_last[1] * (i + 1) * 24 / 24)
                          - error[15, i + 2])
print('sim',sim)
print('sim_err',sim_err)
#2.711455    6.31719745  7.48487487 16.92451262  1.50635407  68109046.83666639
#3.25909828  6.7032152   7.23688396 15.36472613  2.53851773  56730843.94598841
#2.40323089 0.62227532 0.57016057 8.84049268 7.4060548       1.95744138e+07
#2.51908041 0.50939561 0.4825849  8.82174667 7.27930032      1.91582111e+07

x=np.arange(0,16)
y=np.exp(-0.95)
print(y)
from numpy import random
r=np.array([random.uniform(0.8,1.2),random.uniform(0.8,1.2),
          random.uniform(0.8,1.2),random.uniform(0.8,1.2),])
print(r)

parameter51=np.array([[62.04347,0.73623,22.68116],
                      [56.38296,0.76915,22.75318],
                      [61.64586,0.52153,17.21529],
                      [47.65495,0.69649,21.05847],
                      [47.06239,0.75312,19.62496],
                      [41.86888,0.79344,18.94755],
                      [42.96438,0.66429,16.32146],
                      [44.13354,0.60890,15.04451],
                      [33.14947,0.85747,16.45979],
                      [45.17931,0.61195,13.05977],
                      [30.88777,0.84439,14.35511],
                      [32.18740,0.70937,12.27496],
                      [48.96165,0.59233,7.84659],
                      [27.10280,0.87352,10.36760],
                      [31.27481,0.81832,9.09160],
                      [28.84309,0.72287,7.61436]])
step=0.8
amp=0.1
coe_last=np.empty((2))
coe_last[0]=60;coe_last[1]=1

for j in np.arange(0, 100):  # 250000
    i=np.arange(0,16)
    # wg=np.array([0.001,0.00009,0.00012,0.00011])
    wg = np.array([random.uniform(0.02, 0.03), random.uniform(0.003, 0.005)])

    coe_add = coe_last + wg
    y_add = (coe_add[0] * coe_add[1] ** (i) )
    error_add = np.std( parameter51[:,0]- y_add)

    coe_sub = coe_last - wg
    y_sub = (coe_sub[0] * coe_sub[1] ** ( i))
    error_sub = np.std( parameter51[:,0]- y_sub)
    error_e = (error_add - error_sub) * step * wg
    coe_last = coe_last - error_e
print('x',coe_last)
for i in np.arange(0,16):
    y= (coe_add[0] * coe_add[1] ** (i))
    print(y)
r=parameter51[:,2]/parameter51[:,0]

print(r)
print(np.mean(r))
print(np.mean(parameter51[:,1]))

print('mean',np.mean(parameter51[:,1])*2)

parameter20=np.array([52.59914,0.96059,0.38739,0.7189,0.7591,0.5414,
                       0.6764,0.7179,0.7621,0.6642,0.7074,0.8274,0.6699,
                       0.8143,0.7400,0.5923,0.8279,0.8183,0.7208])
print(np.mean(parameter20[3:19])*2)

'''
y_last=(coe_last[0]**2*np.exp(coe_last[1]*(x)/24)+coe_last[2]**2-y**2)**2/w[0,:]

step=0.1
amp=0.1
for i in np.arange(0,10000000):
    wg=0.1

    coe_add=coe_last+wg
    y_add=(coe_add[0]**2*np.exp(coe_add[1]*(x-24)/24)+coe_add[2]**2-y**2)**2/w[0,:]
    error_add=np.std(0-y_add)

    coe_sub=coe_last-wg
    y_sub=(coe_sub[0]**2*np.exp(coe_sub[1]*(x-24)/24)+coe_sub[2]**2-y**2)**2/w[0,:]
    error_sub=np.std(0-y_sub)

    error_e=(error_add-error_sub)*step*wg
    coe_last=coe_last-error_e

#print(error_e)
print(coe_last)
print(np.sqrt(coe_last[0]**2*np.exp(coe_last[1]*(x-24)/24)+coe_last[2]**2))
'''