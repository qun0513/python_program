import pandas as pd
import numpy as np 

# 假设你有一个一维序列，这里我们用一个简单的列表来表示
data = np.loadtxt("D:/vscode/vscode project/年代际预测/PDO_HadISST_1870-2018.txt")
data=np.arange(0,15)

# 将列表转换为pandas的DataFrame
df = pd.DataFrame(data, columns=['Value'])

# 将索引设置为日期范围，这里我们假设数据是按年份排列的
#df.index = pd.date_range(start='2020', periods=len(data), freq='A')

# 计算五年滑动平均
df['5Y_Moving_Average'] = df['Value'].rolling(window=5).mean()

# 输出结果
print(np.savetxt('pdo_rm.txt',df['5Y_Moving_Average']) )