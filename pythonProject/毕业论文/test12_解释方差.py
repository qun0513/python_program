import numpy as np
error=np.loadtxt('D:/毕业设计/error.txt')
#51(48)参数--误差模型
parameter51=np.array([[62.04347,0.73623,22.68116],
                      [56.38296,0.76915,22.75318],
                      [61.64586,0.52153,17.21529],
                      [47.65495,0.69649,21.05847],
                      [47.06239,0.75312,19.62496],#5
                      [41.86888,0.79344,18.94755],
                      [42.96438,0.66429,16.32146],
                      [44.13354,0.60890,15.04451],
                      [33.14947,0.85747,16.45979],
                      [45.17931,0.61195,13.05977],#10
                      [30.88777,0.84439,14.35511],
                      [32.18740,0.70937,12.27496],
                      [48.96165,0.59233,7.84659],
                      [27.10280,0.87352,10.36760],
                      [31.27481,0.81832,9.09160],
                      [28.84309,0.72287,7.61436],
                      [25.00396, 0.72039, 8.20158],
                      [27.43023, 0.78702, 6.77209],
                      [25.00396, 0.72039, 8.20158]])
#35(33)参数--误差模型
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
                      [27.3489,0.7353,0.4348],
                      [25.00396, 0.72039, 8.20158],
                      [27.43023, 0.78702, 6.77209],
                      [25.00396, 0.72039, 8.20158]]) # r=0.36444375
parameter35[:,2]=np.mean(parameter35[:,2])
#20(19)参数--误差模型
parameter20=np.array([56.18,0.957,0.358,0.7266,0.7600,0.5414,
                       0.7048,0.7256,0.7621,0.7008,0.7080,0.7255,0.6699,
                       0.8143,0.7107,0.5923,0.8279,0.7249,0.7166,0.72039,
                      0.78702,0.72039])
#4参数--误差模型
parameter4=np.array([57.23389881,0.95465204,0.72303738,0.35315367])
print(parameter51)
print(parameter35)
print(parameter20)
print(parameter4)

#y51=np.loadtxt('D:/毕业设计/simulate_perceive.txt')
y51=y35=np.empty((19,5));y20=np.empty((19,5));y4=np.empty((19,5))
for i in np.arange(0,19):
    for j in np.arange(0,5):
        if j<2:
            y51[i,j]=np.sqrt(parameter51[i,0]**2*np.exp(parameter51[i,1]*(j+3)*12/24)
                                   +parameter51[i,2]**2)

            y35[i,j]=np.sqrt(parameter35[i,0]**2*np.exp(parameter35[i,1]*(j+3)*12/24)
                       +parameter35[i,2]*parameter35[i,0]**2)

            y20[i,j]=np.sqrt(parameter20[0]**2*parameter20[1]**(i*2)*\
                     np.exp(parameter20[3+i]*(j+3)*12/24)+\
                     parameter20[2]*parameter20[0]**2*parameter20[1]**(i*2))

            y4[i,j]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+3)*12/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))

        if j>=2:
            y51[i,j]=np.sqrt(parameter51[i, 0] ** 2 * np.exp(parameter51[i, 1] * (j + 1) * 24 / 24)
                    + parameter51[i, 2] ** 2)
            y35[i,j]=np.sqrt(parameter35[i,0]**2*np.exp(parameter35[i,1]*(j+1)*24/24)
                       +parameter35[i,2]*parameter35[i,0]**2)

            y20[i,j]=np.sqrt(parameter20[0]**2*parameter20[1]**(i*2)*\
                     np.exp(parameter20[3+i]*(j+1)*24/24)+\
                     parameter20[2]*parameter20[0]**2*parameter20[1]**(i*2))

            y4[i,j]=np.sqrt(parameter4[0]**2*parameter4[1]**(2*i)*
                       np.exp(parameter4[2]*(j+1)*24/24)
                       +parameter4[3]*parameter4[0]**2*parameter4[1]**(2*i))


print('y51','\n',y51)
print('y35','\n',y35)
print('y20','\n',y20)
print('y4',"\n",y4)

#残差平方和
y51_residual=0;y35_residual=0;y20_residual=0;y4_residual=0
y_model=0
error_mean=np.mean(error[:,2::])
print('error_mean','\n',error_mean)
for i in np.arange(0,12):
    for j in np.arange(0,5):
        y51_residual=y51_residual+(y51[i,j]-error[i,j+2])**2
        y35_residual=y35_residual+(y35[i,j]-error[i,j+2])**2
        y20_residual=y20_residual+(y20[i,j]-error[i,j+2])**2
        y4_residual =y4_residual+(y4[i,j]-error[i,j+2])**2

y51_r=0;y35_r=0;y20_r=0;y4_r=0
for i in [12,13,15,16,17,18]:
    for j in np.arange(0,5):
        y51_r=y51_r+(y51[i,j]-error[i,j+2])**2
        y35_r=y35_r+(y35[i,j]-error[i,j+2])**2
        y20_r=y20_r+(y20[i,j]-error[i,j+2])**2
        y4_r =y4_r+(y4[i,j]-error[i,j+2])**2


#方差
var51=np.var(y51)
var35=np.var(y35)
var20=np.var(y20)
var4 =np.var(y4)

var0=np.var(error[0:12,2::])
var11=np.var(error[[12,13,15,16,17,18],2::])

#R2
var_exp51=1-y51_residual/(var51*60)
var_exp35=1-y35_residual/(var35*60)
var_exp20=1-y20_residual/(var20*60)
var_exp4 =1-y4_residual/(var4*60)

var_exp51_0=1-y51_residual/(var0*60)  #right
var_exp35_0=1-y35_residual/(var0*60)
var_exp20_0=1-y20_residual/(var0*60)
var_exp4_0 =1-y4_residual/(var0*60)

var_exp51_1=1-y51_r/(var11*30)  #right
var_exp35_1=1-y35_r/(var11*30)
var_exp20_1=1-y20_r/(var11*30)
var_exp4_1 =1-y4_r/(var11*30)

print('R2')
print(var_exp51,var_exp51_0,var_exp51_1)
print(var_exp35,var_exp35_0,var_exp35_1)
print(var_exp20,var_exp20_0,var_exp20_1)
print(var_exp4,var_exp4_0,var_exp4_1)

#RMSE
var_exp51_1=np.sqrt(y51_residual/80)
var_exp35_1=np.sqrt(y35_residual/80)
var_exp20_1=np.sqrt(y20_residual/80)
var_exp4_1=np.sqrt(y4_residual/80)

print('RMSE')
print(var_exp51_1)
print(var_exp35_1)
print(var_exp20_1)
print(var_exp4_1)

'''
from sklearn.metrics import r2_score

r51=r2_score(y51,error[:,2::])
r35=r2_score(y35,error[:,2::])
r20=r2_score(y20,error[:,2::])
r4=r2_score(y4,error[:,2::])
#print(error[:,2::])
print('r2_score')
print(r51)
print(r35)
print(r20)
print(r4)
'''

