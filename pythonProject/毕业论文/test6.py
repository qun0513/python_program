import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt

error=np.loadtxt('D:/毕业设计/error.txt')
parameter4=np.array([64.2473877,0.93596054])

yanalysis=np.empty((5))
for i in np.arange(0,5):
    yanalysis[i]=parameter4[0]*parameter4[1]**(i)
y1=yanalysis[::-1]


from scipy.optimize import curve_fit
fig=plt.figure()
ax1=fig.add_subplot(111)



def nihe(x,a,b):
    return a*np.exp(b*x/24)

simulate=np.empty((16,120))
x11=np.arange(0,120)
xx=np.array([60,56,59,51,50,50,46,51,36,51,37,40,57,32,39,37])
xy=np.array([99,100,85,92.5,95.1,93.2,87.2,84,94,84.8,90.5,83,80.1,88.7,89,80.5])
for i in np.arange(0,16):
    x=np.array([36,48,72,96,120])
    y=error[i,2::]                      #感知误差
    popt,pcov=curve_fit(nihe,x,y,maxfev = 1000000)  #p0=(1,1,1)
    #print(popt)
    #x=np.array(x)

    simulate[i] = (popt[0]*np.exp(popt[1]*x11/24))

    #plt.plot(np.linspace(xx[i], xy[i], 120), simulate[i])
    #plt.plot(x11, simulate[i, :])
    if 15>i>1:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                 np.linspace(xx[i], xy[i], 120)[47],
                 np.linspace(xx[i], xy[i], 120)[71],
                 np.linspace(xx[i], xy[i], 120)[95],
                 np.linspace(xx[i], xy[i], 120)[119],
                 ],error[i,2::],c='k',marker='*')
    if i==0:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='r', marker='*')
    if i==15:
        plt.scatter([np.linspace(xx[i], xy[i], 120)[35],
                     np.linspace(xx[i], xy[i], 120)[47],
                     np.linspace(xx[i], xy[i], 120)[71],
                     np.linspace(xx[i], xy[i], 120)[95],
                     np.linspace(xx[i], xy[i], 120)[119],
                     ], error[i, 2::], c='b', marker='*')


#print(simulate)
y2016 = 64.2473877 * 0.93596054 ** 16
#print(y2016)

yy=np.empty((100))
for i in np.arange(0,100):
    yy[i]=7.49/0.96196054**(i)
print(yy)

x22=np.arange(0,100)
plt.plot(x22,yy,c='r',lw=2)
plt.scatter(0,yy[0],s=50)
plt.scatter(33,yy[33],s=50,c='b')
plt.scatter(50,yy[50],s=50,c='r')

#ax2=fig.add_subplot(122)
xx=np.array([60,56,59,51,50,50,46,51,36,51,37,40,57,32,39,37])
#plt.plot(np.linspace(xx[0],100,120),simulate[0,:])
plt.show()
#y2050=64.2473877*0.93596054**50
#print(y2050)

#print(simulate)
ysim=simulate[:,0]
print(ysim)
ysim1=simulate[:,119]
print(ysim1)
#print(y)

#fig,ax=plt.subplots()
#ax = skplt.metrics.plot_lift_curve(y1, simulate)
#skplt.metrics.plot_lift_curve(y1, simulate, ax=ax)
#plt.plot(x1,simulate[0,:])
#plt.show()