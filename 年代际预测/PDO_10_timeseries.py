import numpy as np
import os
import glob
from scipy.stats import pearsonr
#from sklearn.preprocessing import scale

yr=np.arange(1950,2010,10)
ts=[]
for i in np.arange(4,5):
    year=yr[i]
    timeseries=np.loadtxt(f"D:/decadal prediction/results/PDOindex_HadISST_{year}-{year+10}.txt")
    #ts.append(timeseries)
    ts=np.concatenate((ts,timeseries))
pdo=np.loadtxt("D:/decadal prediction/results/PDOindex_HadISST1950-2018.txt")
print(ts.shape,pdo.shape)
pdo=pdo[480:600]


'''
pdo1=np.empty((660))
ts1=np.empty((660))
print(ts,'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print(pdo)
for i in np.arange(30, len(pdo)-30):
    pdo1[i-30]=np.mean(pdo[i-30: i+30])
    ts1[i-30]=np.mean(ts[i-30: i+30])
'''

r,p=pearsonr(ts,pdo)
print(r,p)