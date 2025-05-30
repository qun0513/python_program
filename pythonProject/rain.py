import numpy as np
from pandas import DataFrame
#a=open('D:/PY/rain2.txt')
b=np.loadtxt('D:/PY/rain2.txt')
c=DataFrame(b)
print(c)
for i in np.arange(0,100):
    if 0.1<c.loc[i,2]<10:
        c.loc[i,2]='小雨'
    elif 10<=b[i,2]<25:
        c.loc[i,2]='中雨'
    elif 25<=b[i,2]<50:
        c.loc[i,2]='大雨'
    elif 50<=b[i,2]<100:
        c.loc[i,2]='暴雨'
    elif 100<=b[i,2]<250:
        c.loc[i,2]='大暴雨'
    elif 250<b[i,2]:
        c.loc[i,2]='特大暴雨'
d=c.values
#print(c)
print(d)
'''
r2 = open("D:/PY/rain2.txt")
r2.readline()
level = []
a = []
c = []
for line in r2:
    station = int(line.split()[0])
    month = int(line.split()[1])
    rain = float(line.split()[2])
    if 0.1 <= rain < 10 and 100 <= station <= 105:
        level.extend([station, month, '小雨'])
    elif 10 <= rain < 25 and 100 <= station <= 105:
        level.extend([station, month, '中雨'])
    elif 25 <= rain < 50 and 100 <= station <= 105:
        level.extend([station, month, '大雨'])
    elif 50 <= rain < 100 and 100 <= station <= 105:
        level.extend([station, month, '暴雨'])
    elif 100 <= rain < 250 and 100 <= station <= 105:
        level.extend([station, month, '大暴雨'])
    elif rain >= 250 and 100 <= station <= 105:
        level.extend([station, month, '特大暴雨'])
print(level)
'''

'''

'''