import numpy as np
import math
ah=np.loadtxt('D:/ZD/DATA/555/H500.txt')
bh=np.loadtxt('D:/ZD/DATA/555/H850.txt')
at=np.loadtxt('D:/ZD/DATA/555/T500.txt')
bt=np.loadtxt('D:/ZD/DATA/555/T850.txt')
au=np.loadtxt('D:/ZD/DATA/555/U500.txt')
bu=np.loadtxt('D:/ZD/DATA/555/U850.txt')
av=np.loadtxt('D:/ZD/DATA/555/V500.txt')
bv=np.loadtxt('D:/ZD/DATA/555/V850.txt')
p=3.14;a=6370;d=400;rd=0.00976;R=287;g=9.8
d_lat=(d/a)*p/180
a_theta=np.empty((11,10))
b_theta=np.empty((11,10))
a_sigma=np.empty((11,10))
b_sigma=np.empty((11,10))
a_xi=np.empty((11,10))
b_xi=np.empty((11,10))
f=np.empty((11,10))
a_vv=np.empty((11,10))
b_vv=np.empty((11,10))
th=np.empty((11,10))
dth=np.empty((11,10))

for i in np.arange(0,11):
  for j in np.arange(0,10):
    lat=60-i*d_lat
    f[i,j]=2*(2*p/86400)*math.sin(lat)
    r=-(at[i,j]-bt[i,j])/(ah[i,j]-bh[i,j])
    a_theta[i,j]=at[i,j]*(1000/500)**0.286
    b_theta[i,j]=bt[i,j]*(1000/850)**0.286
    a_sigma[i,j]=-(R*at[i,j]/500/a_theta[i,j])*((a_theta[i,j]-b_theta[i,j])/(500-850))
    b_sigma[i,j]=-(R*bt[i,j]/850/b_theta[i,j])*((a_theta[i,j]-b_theta[i,j])/(500-850))
sigma=(a_sigma+b_sigma)/2
#for i in np.arange(0,11):
 #   for i in np.arange(0,10):
#求涡度  a_xi
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北  中央差分
          for k in np.arange(1,9):
            a_xi[i,k]=(av[i,k+1]-av[i,k-1])/2/d-(au[0,k]-au[1,k])/d  # /
        if i==10:    #南
          for k in np.arange(1,9):
            a_xi[i,k]=(av[i,k-1]-av[i,k+1])/2/d-(au[i-1,k]+au[i,k])/d
        if j==0:     #东
          for l in np.arange(1,10):
            a_xi[l,j]=(av[l,1]-av[l,0])/d-(au[l-1,j]-au[l+1,j])/2/d
        if j==9:     #西
          for m in np.arange(1,10):
            a_xi[m,j]=(av[m,j]-av[m,j-1])/d-(au[m-1,j]-au[m+1,j])/2/d
for i in np.arange(1,10):
    for j in np.arange(1,9):
            a_xi[i,j]=(av[i,j+1]-av[i,j-1]-au[i-1,j]+au[i+1,j])/2/d
a_xi[0,0]=(a_xi[0,1]+a_xi[1,0])/2
a_xi[0,9]=(a_xi[0,8]+a_xi[1,9])/2
a_xi[10,0]=(a_xi[9,0]+a_xi[10,1])/2
a_xi[10,9]=(a_xi[10,8]+a_xi[9,9])/2
print(a_xi)
#求涡度平流项  a_vv
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            a_vv[i,k]=au[i,j]*(a_xi[i,k+1]-a_xi[i,k-1])/2/d-av[i,j]*(
                    a_xi[0,k]+f[0,k]-a_xi[1,k]-f[1,k])/2/d
        if i==10:    #南
          for k in np.arange(1,9):
            a_vv[i,k]=au[i,j]*(a_xi[i,k-1]-a_xi[i,k+1])/2/d-av[i,j]*(
                    a_xi[i-1,k]+f[i-1,k]-a_xi[i,k]-f[i,k])/d  ###zheng fu hao
        if j==0:     #东
          for l in np.arange(1,10):
            a_vv[l,j]=au[i,j]*(a_xi[l,1]-a_xi[l,0])/d-av[i,j]*(
                    a_xi[l-1,j]+f[l-1,j]-a_xi[l+1,j]-f[l+1,j])/2*d
        if j==9:     #西
          for m in np.arange(1,10):
            a_vv[m,j]=au[i,j]*(a_xi[m,j]-a_xi[m,j-1])/d-av[i,j]*(
                    a_xi[m-1,j]+f[m-1,j]-a_xi[m+1,j]-f[m+1,j])/2/d
for i in np.arange(1,10):
    for j in np.arange(1,9):
        a_vv[i,j]=au[i,j]*(a_xi[i,j+1]-a_xi[i,j-1])/2/d+av[i,j]*(
                                                                a_xi[i,j+1]+f[i,j+1]
                                                               -a_xi[i,j-1]-f[i,j-1]
                                                               )/2/d
a_vv[0,0]=(a_vv[0,1]+a_vv[1,0])/2
a_vv[0,9]=(a_vv[0,8]+a_vv[1,9])/2
a_vv[10,0]=(a_vv[9,0]+a_vv[10,1])/2
a_vv[10,9]=(a_vv[10,8]+a_vv[9,9])/2

#求涡度  b_xi
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            b_xi[i,k]=(bv[i,k+1]-bv[i,k-1])/2/d-(bu[0,k]-bu[1,k])/d
        if i==10:    #南
          for k in np.arange(1,9):
            b_xi[i,k]=(bv[i,k-1]-bv[i,k+1])/2/d-(bu[i-1,k]+bu[i,k])/d
        if j==0:     #东
          for l in np.arange(1,10):
            b_xi[l,j]=(bv[l,1]-bv[l,0])/d-(bu[l-1,j]-bu[l+1,j])/2/d
        if j==9:     #西
          for m in np.arange(1,10):
            b_xi[m,j]=(bv[m,j]-bv[m,j-1])/d-(bu[m-1,j]-bu[m+1,j])/2/d
for i in np.arange(1,10):
    for j in np.arange(1,9):
            b_xi[i,j]=(bv[i,j+1]-bv[i,j-1]-bu[i-1,j]+bu[i+1,j])/2/d
b_xi[0,0]=(b_xi[0,1]+b_xi[1,0])/2
b_xi[0,9]=(b_xi[0,8]+b_xi[1,9])/2
b_xi[10,0]=(b_xi[9,0]+b_xi[10,1])/2
b_xi[10,9]=(b_xi[10,8]+b_xi[9,9])/2
print(b_xi)
#求涡度平流项  b_vv
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            b_vv[i,k]=bu[i,j]*(b_xi[i,k+1]-b_xi[i,k-1])/2/d-bv[i,j]*(
                    b_xi[0,k]+f[0,k]-b_xi[1,k]-f[1,k])/2/d
        if i==10:    #南
          for k in np.arange(1,9):
            b_vv[i,k]=bu[i,j]*(b_xi[i,k-1]-b_xi[i,k+1])/2/d-bv[i,j]*(
                    b_xi[i-1,k]+f[i-1,k]-b_xi[i,k]-f[i,k])/d  ###zheng fu hao
        if j==0:     #东
          for l in np.arange(1,10):
            b_vv[l,j]=bu[i,j]*(b_xi[l,1]-b_xi[l,0])/d-bv[i,j]*(
                    b_xi[l-1,j]+f[l-1,j]-b_xi[l+1,j]-f[l+1,j])/2/d
        if j==9:     #西
          for m in np.arange(1,10):
            b_vv[m,j]=bu[i,j]*(b_xi[m,j]-b_xi[m,j-1])/d-bv[i,j]*(
                    b_xi[m-1,j]+f[m-1,j]-b_xi[m+1,j]-f[m+1,j])/2/d
for i in np.arange(1,10):
    for j in np.arange(1,9):
        b_vv[i,j]=bu[i,j]*(b_xi[i,j+1]-b_xi[i,j-1])/2/d+bv[i,j]*(
                                                                b_xi[i,j+1]+f[i,j+1]
                                                               -b_xi[i,j-1]-f[i,j-1]
                                                               )/2/d

b_vv[0,0]=(b_vv[0,1]+b_vv[1,0])/2
b_vv[0,9]=(b_vv[0,8]+b_vv[1,9])/2
b_vv[10,0]=(b_vv[9,0]+b_vv[10,1])/2
b_vv[10,9]=(b_vv[10,8]+b_vv[9,9])/2
print(b_vv)

first1=(f/sigma)*((a_vv-b_vv)/(500-850))
print('1','\n',first1)  #第一项

#########################################################################################
d_f=ah-bh
print(d_f)
dfp=d_f/(500-850)

#求温度平流项随高度变化项 th
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            th[i,k]=au[i,j]*(dfp[i,k+1]-dfp[i,k-1])/2/d-av[i,j]*(
                    dfp[0,k]+f[0,k]-dfp[1,k]-f[1,k])/2/d
        if i==10:    #南
          for k in np.arange(1,9):
            th[i,k]=au[i,j]*(dfp[i,k-1]-dfp[i,k+1])/2/d-av[i,j]*(
                    dfp[i-1,k]+f[i-1,k]-dfp[i,k]-f[i,k])/d  ###zheng fu hao
        if j==0:     #东
          for l in np.arange(1,10):
            th[l,j]=au[i,j]*(dfp[l,1]-dfp[l,0])/d-av[i,j]*(
                    dfp[l-1,j]+f[l-1,j]-dfp[l+1,j]-f[l+1,j])/2/d
        if j==9:     #西
          for m in np.arange(1,10):
            th[m,j]=au[i,j]*(dfp[m,j]-dfp[m,j-1])/d-av[i,j]*(
                    dfp[m-1,j]+f[m-1,j]-dfp[m+1,j]-f[m+1,j])/2/d
for i in np.arange(1,10):
    for j in np.arange(1,9):
        th[i,j]=au[i,j]*(dfp[i,j+1]-dfp[i,j-1])/2/d+av[i,j]*(
                                                                dfp[i,j+1]+f[i,j+1]
                                                               -dfp[i,j-1]-f[i,j-1]
                                                               )/2/d
print('th','\n',th)

#求th的laplace ~ dth
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            dth[i,k]=(th[i,k-1]+th[i,k+1]+th[i+1,k]-3*th[i,k])/d**2
        if i==10:    #南
          for k in np.arange(1,9):
            dth[i,k]=(th[i,k-1]+th[i,k+1]+th[i-1,k]-3*th[i,k])/d**2
        if j==0:     #东
          for l in np.arange(1,10):
            dth[l,j]=(th[l,1]+th[l-1,0]+th[l+1,0]-3*th[l,j])/d**2
        if j==9:     #西
          for m in np.arange(1,10):
            dth[m,j]=(th[m-1,j]+th[m,j-1]+th[m+1,j]-3*th[m,j])/d**2
for i in np.arange(1,10):
    for j in np.arange(1,9):
            dth[i,j]=(th[i-1,j]+th[i+1,j]+th[i,j-1]+th[i,j+1]-4*th[i,j])/d**2
dth[0,0]=(th[0,1]+th[1,0])/d**2
dth[0,9]=(th[0,8]+th[1,9])/d**2
dth[10,0]=(th[9,0]+th[10,1])/d**2
dth[10,9]=(th[10,8]+th[9,9])/d**2

dth2=dth/sigma
print(dth2)

F=d**2*(first1+dth2)
lambda0=2*f*(f+(a_xi+b_xi)/2)/(675*(675-1000)*(a_sigma+b_sigma)/2)
ww=np.zeros((11,10))
'''
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            a_xi[i,k]=(av[0,k+1]-av[1,k-1])/2/d-(au[0,k]-au[1,k])/2/d  # /
        if i==10:    #南
          for k in np.arange(1,9):
            a_xi[i,k]=(av[i,k-1]-av[i,k+1])/2/d-(au[i-1,k]+au[i,k])/d
        if j==0:     #东
          for l in np.arange(1,10):
            a_xi[l,j]=(av[l,1]-av[l,0])/d-(au[l-1,j]-au[l+1,j])/2/d
        if j==9:     #西
          for m in np.arange(1,10):
            a_xi[m,j]=(av[m,j]-av[m,j-1])/d-(au[m-1,j]-au[m+1,j])/2/d
for i in np.arange(1,10):
    for j in np.arange(1,9):
            a_xi[i,j]=(av[i,j+1]-av[i,j-1]-au[i-1,j]+au[i+1,j])/2/d
a_xi[0,0]=(a_xi[0,1]+a_xi[1,0])/2
a_xi[0,9]=(a_xi[0,8]+a_xi[1,9])/2
a_xi[10,0]=(a_xi[9,0]+a_xi[10,1])/2
a_xi[10,9]=(a_xi[10,8]+a_xi[9,9])/2
'''





'''
#求th的laplace ~ dth
for i in np.arange(0,11):
    for j in np.arange(0,10):
        if i==0:     #北
          for k in np.arange(1,9):
            dth[i,k]=(th[0,k+1]-th[1,k-1])/2*d-(th[0,k]-th[1,k])/2*d
        if i==10:    #南
          for k in np.arange(1,9):
            dth[i,k]=(th[i,k-1]-th[i,k+1])/2*d-(th[i-1,k]+th[i,k])/d
        if j==0:     #东
          for l in np.arange(1,10):
            dth[l,j]=(th[l,1]-th[l,0])/d-(th[l-1,j]-th[l+1,j])/2*d
        if j==9:     #西
          for m in np.arange(1,10):
            dth[m,j]=(th[m,j]-th[m,j-1])/d-(th[m-1,j]-th[m+1,j])/2*d
for i in np.arange(1,10):
    for j in np.arange(1,9):
            dth[i,j]=(th[i-1,j]+th[i+1,j]+th[i,j-1]+th[i,j+1]-4*th[i,j])/d**2
b_xi[0,0]=(b_xi[0,1]+b_xi[1,0])/2
b_xi[0,9]=(b_xi[0,8]+b_xi[1,9])/2
b_xi[10,0]=(b_xi[9,0]+b_xi[10,1])/2
b_xi[10,9]=(b_xi[10,8]+b_xi[9,9])/2
'''


