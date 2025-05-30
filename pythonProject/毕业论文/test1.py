import numpy as np
error=np.loadtxt('/home/dell/ZQ19/bysj/error.txt')
sem=np.loadtxt('/home/dell/ZQ19/bysj/sem.txt')
sem2=sem*2.18
w=np.empty((16,5))
for i in np.arange(0,16):
    for k in np.arange(2, 7):
        w[i,k-2]=sem[i,k]/sum(sem[i,2::])

coe_last=np.zeros((3))
coe_start=np.zeros((3))
coe_int=np.zeros((3))

coe_start[0]=15;coe_start[1]=0.5;coe_start[2]=0           #初值
coe_int[0]=0.1;coe_int[1]=0.01;coe_int[2]=0.001         #间隔

sim=np.empty((16,5))
sim_err=np.empty((16,5))
xbest=np.zeros((16,3))
simmin=1e20

#for i in np.arange(0,10000):
#    coe_last[0]=coe_start[0]+coe_int[0]*i
#    coe_last[1]=coe_last[0]*0.02+coe_int[1]*i

for i in np.arange(0,650):
    coe_last[0]=coe_start[0]+coe_int[0]*i
    for j in np.arange(0,130):
        coe_last[1]=coe_start[1]+coe_int[1]*j
        for k in np.arange(0,800):
            coe_last[2]=coe_start[2]+coe_int[2]*k
            coe_last[2]=coe_last[2]*coe_last[0]
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

with open('/home/dell/ZQ19/bysj/xbest1.txt','w') as outfile:
    np.savetxt(outfile,xbest)
with open('/home/dell/ZQ19/bysj/ysimulate1.txt','w') as outfile:
    np.savetxt(outfile,ysimulate)