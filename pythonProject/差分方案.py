import numpy as np
import math
#import sys
#np.set_printoptions(threshold=sys.maxsize)
dx=0.01;dt=0.004;p=3.14;c=2
uxt=np.empty((1001,101))
'''
r=np.empty((1001,101))
s=np.empty((1001,101))
t=np.empty((1001,101))
'''
#固定边界条件
for j in np.arange(0,101):
        uxt[1,j]=math.sin(2*p*(j-1)*dx)+2

for i in np.arange(0,1001):
        uxt[i,0]=uxt[0,0]

for i in np.arange(1,1001):
        for  j in np.arange(1,101):
            uxt[i,j]=uxt[i-1,j]-c*dt/dx*(uxt[i-1,j]-uxt[i-1,j-1])

S=[]
for i in np.arange(0,1001):
    s=0
    for j in np.arange(0,101):
        s=s+0.5*uxt[i,j]**2
    S.append(s)
ss=np.array(S)


#周期边界条件
for j in np.arange(0,101):
        uxt[1,j]=math.sin(2*p*(j-1)*dx)+2.0

for i in np.arange(1,1001):
    for j in np.arange(1,101):
            uxt[i,j]=uxt[i-1,j]-c*dt/dx*(uxt[i-1,j]-uxt[i-1,j-1])
            if i<1000:
              uxt[i+1,1]=uxt[i,100]

T=[]
for i in np.arange(0,1001):
    t=0
    for j in np.arange(0,101):
        t=t+0.5*uxt[i,j]**2
    T.append(t)
tt = np.array(T)

#相邻边界条件
for j in np.arange(0,101):
        uxt[1,]=math.sin(2*p*(j-1)*dx)+2.0

for i in np.arange(1,101):
    for j in np.arange(1,101):
            uxt[i,j]=uxt[i-1,j]-c*dt/dx*(uxt[i-1,j]-uxt[i-1,j-1])
            uxt[i,1]=uxt[i,2]
R=[]
for i in np.arange(0,1001):
    r=0
    for j  in np.arange(0,101):
        r=r+0.5*uxt[i,j]**2
    R.append(r)
rr = np.array(R)

print(ss)
print(ss.max)
print(ss.min)
print(tt)
print(rr)
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
x=np.arange(1,1002)
ax.plot(x,rr)
plt.show()


'''
program CFFA
    integer::i,j,c=2
    real::dx=0.01,dt=0.004,p=3.14,uxt(1001,101)
    !固定边界条件
    do j=1,101
        uxt(1,j)=sin(2*p*(j-1)*dx)+2.0
    end do
    do i=1,1001
        uxt(i,1)=uxt(1,1)
    end do
    do i=2,1001
        do j=2,101
            uxt(i,j)=uxt(i-1,j)-c*dt/dx*(uxt(i-1,j)-uxt(i-1,j-1))
        end do
    end do
    do i=1,1001
        s=0
        do j=1,101
            s=s+0.5*uxt(i,j)**2
        end do
        print*,s
    end do
    print*,111
    !周期边界条件
    do j=1,101
        uxt(1,j)=sin(2*p*(j-1)*dx)+2.0
    end do
    do i=2,1001
        do j=2,101
            uxt(i,j)=uxt(i-1,j)-c*dt/dx*(uxt(i-1,j)-uxt(i-1,j-1))
            uxt(i+1,1)=uxt(i,101)
        end do
    end do
    do i=1,1001
        t=0
        do j=1,101
            t=t+0.5*uxt(i,j)**2
        end do
        print*,t
    end do
    print*,t
    !相邻边界条件
    do j=1,101
        uxt(1,j)=sin(2*p*(j-1)*dx)+2.0
    end do
    do i=2,1001
        do j=2,101
            uxt(i,j)=uxt(i-1,j)-c*dt/dx*(uxt(i-1,j)-uxt(i-1,j-1))
            uxt(i,1)=uxt(i,2)
        end do
    end do
    do i=1,1001
        r=0
        do j=1,101
            r=r+0.5*uxt(i,j)**2
        end do
        print*,r
    end do
    print*,r
end program
'''
