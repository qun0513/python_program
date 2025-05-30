import numpy as np
import pandas as pd
from datetime import datetime

def load_and_add_time_index(txt_path, start_year=1700):
#Pandas使用的时间戳是基于纳秒的，所以最大范围是从大约1677年到2262年.
    data = np.loadtxt(txt_path)
    #data=data[288:1332]
    # 计算年份和月份
    n = len(data)
    months = np.arange(n) % 12 + 1  # 月份从1到12
    years = start_year + (np.arange(n) // 12)  # 每12个月年份加1
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'year': years,
        'month': months,
        'ssta': data
    })
    
    # 生成日期列（假设每月数据为当月第一天）
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str)+'-1' , format='%Y-%m-%d')
    
    # 按日期排序
    df = df.sort_values('date')
    
    return df     #df[['year', 'month', 'date', 'ssta']]


def detect_events(df, condition, end_condition, threshold, min_consecutive, event_type):
    """
    （此函数与之前版本相同，为方便阅读保留核心注释）
    事件检测核心函数
    参数：
        df: 输入DataFrame（必须包含'ssta'和'date'列）
        condition: 事件开始条件函数（如lambda x: x > 0）
        end_condition: 事件结束条件函数（如lambda x: x < 0）
        threshold: 有效事件阈值（暖事件0.25，冷事件-0.25）
        min_consecutive: 最小持续月数（5）
        event_type: 事件类型标识（'warm'/'cold'）
    返回：
        事件字典列表，包含事件详细信息
    """
    events = []
    in_event = False
    start_idx = -1
    
    for idx, row in df.iterrows():
        ssta = row['ssta']
        
        # 检测事件开始
        if condition(ssta):
            if not in_event:
                start_idx = idx         #?---
                in_event = True
        # 检测事件结束
        else:
            if in_event and end_condition(ssta):
                end_idx = idx - 1     #?---
                
                if end_idx >= start_idx:
                    subset = df.iloc[start_idx:end_idx+1]     #?-----
                    
                    # 检查连续达标月数
                    consecutive = 0
                    max_consecutive = 0
                    for val in subset['ssta']:
                        if (event_type == 'warm' and val > threshold) or \
                           (event_type == 'cold' and val < threshold):
                            consecutive += 1
                            max_consecutive = max(max_consecutive, consecutive)
                        else:
                            consecutive = 0
                    
                    # 满足最小持续时间要求
                    if max_consecutive >= min_consecutive:
                        # 计算事件详细信息-----
                        start_date = subset.iloc[0]['date']
                        end_date = subset.iloc[-1]['date']   #最后一行
                        duration = (end_date.year - start_date.year) * 12 + \
                                    (end_date.month - start_date.month) + 1
                        
                        # 获取极值及其月份
                        if event_type == 'warm':
                            extreme = subset['ssta'].max()
                            ext_months = subset[subset['ssta'] == extreme][['year', 'month']].values.tolist()
                        else:
                            extreme = subset['ssta'].min()
                            ext_months = subset[subset['ssta'] == extreme][['year', 'month']].values.tolist()
                        
                        events.append({
                            'type': '暖事件' if event_type == 'warm' else '冷事件',
                            'start': start_date.strftime('%Y-%m'),
                            'end': end_date.strftime('%Y-%m'),
                            'duration': duration,
                            'extreme': round(extreme, 2),
                            'ext_months': [f"{y}-{m:02d}" for y, m in ext_months]
                        })
                in_event = False
    return events




if __name__ == "__main__":
    txt_path = 'D:/decadal prediction/results/piControl/reanalysis/koe_index_1980-2018_HadISST.txt'
    df = load_and_add_time_index(txt_path, start_year=1700)
    koe_idx=np.loadtxt(txt_path) #observation
    std=np.std(koe_idx)
    print(std/2)
    
    # 事件检测（参数与之前相同）
    warm_events = detect_events(df, 
        condition=lambda x: x > 0.25,
        end_condition=lambda x: x < 0.25,
        threshold=std/2,  #? -----
        min_consecutive=5,
        event_type='warm'
    )
    
    cold_events = detect_events(df,
        condition=lambda x: x < -0.25,
        end_condition=lambda x: x >= -0.25,
        threshold=-std/2,
        min_consecutive=5,
        event_type='cold'
    )

    all_events = warm_events + cold_events
    #print(warm_events)
    # 筛选典型事件（极值出现在7-9月）
    typical_events = []
    for event in all_events:
        for month_str in event['ext_months']:
            month = int(month_str.split('-')[1]) #月份列
            if 7 <= month <= 9:
                typical_events.append(event)
                break  # 只要有一个极值月份符合条件即可
    

    '''# 结果输出示例
    print(f"共检测到{len(all_events)}个事件")
    print("\n所有事件列表：")
    for event in all_events:
        print(f"{event['type']} {event['start']}~{event['end']} "
              f"持续{event['duration']}个月，极值{event['extreme']} "
              f"出现在：{', '.join(event['ext_months'])}")
    
    print(f"共检测到{len(typical_events)}个典型事件")
    print("\n典型事件（极值在夏季）：")
    for event in typical_events:
        print(f"{event['type']} {event['start']}~{event['end']} "
              f"极值月份：{', '.join(event['ext_months'])}")'''

    

    from collections import Counter
    #极值出现的季节的统计
    # 定义季节分类规则
    seasons = {
        1: '冬季', 2: '冬季', 3: '冬季',
        4: '春季', 5: '春季', 6: '春季',
        7: '夏季', 8: '夏季', 9: '夏季',
        10: '秋季', 11: '秋季', 12: '秋季'
    }

    # 初始化月份和季节统计字典
    month_counts = Counter()
    season_counts = Counter()

    duration=[]
    extreme=[]
    # 遍历 events 列表
    for event in warm_events:  #?---
        # 提取 ext_months 列
        ext_months = event['ext_months']
        duration.append(event['duration'])
        extreme.append(event['extreme'])
        
        # 遍历 ext_months，提取月份并分类
        for month_str in ext_months:
            year, month = map(int, month_str.split('-'))  # 将字符串转换为年和月
            month_counts[month] += 1  # 更新月份统计
            season = seasons[month]      # 根据月份获取季节
            season_counts[season] += 1  # 更新季节统计
    
    #print(type(warm_events))
    


    # 打印月份统计结果
    print("月份统计结果：")
    for month in range(1, 13):  # 从1到12月
        print(f"{month}月: {month_counts[month]}次")

    # 打印季节统计结果
    print(f"warm_events季节统计结果：") #?---
    for season in ['冬季', '春季', '夏季', '秋季']:
        print(f"{season}: {season_counts[season]}次")
    
    print(np.mean(duration))
    print(np.mean(extreme))

#筛选典型事件的年份

#print(df)


#? 挑选koe冷暖事件----------------
'''
def find_koe_events(koe_idx, threshold=0.25, min_duration=5, event_type='warm'):
    """
    筛选出符合KOE-related SST warm或cold event条件的事件。
    
    参数:
    koe_idx (numpy.ndarray): KOE-related SSTA的时间序列，形状为 (time,)
    threshold (float): SSTA的阈值，默认为0.25°C
    min_duration (int): 持续时间的最小月数，默认为5个月
    event_type (str): 事件类型，'warm' 表示暖事件，'cold' 表示冷事件，默认为 'warm'
    
    返回:
    list: 符合条件的事件列表，每个事件包含起始索引、结束索引和持续月数
    """
    events = []
    current_start = None
    current_duration = 0

    # 根据事件类型调整阈值和比较逻辑
    if event_type == 'warm':
        condition = lambda value: value > threshold
    elif event_type == 'cold':
        condition = lambda value: value < -threshold
    else:
        raise ValueError("event_type must be 'warm' or 'cold'")

    for i, value in enumerate(koe_idx):
        # 检查当前值是否满足条件
        if condition(value):
            if current_start is None:
                # 开始一个新的事件
                current_start = i
                current_duration = 1
            else:
                # 延续当前事件
                current_duration += 1
        else:
            # 当前值不满足条件，检查是否需要记录事件
            if current_start is not None and current_duration >= min_duration:
                events.append({
                    'start_index': current_start,
                    'end_index': i - 1,
                    'duration': current_duration
                })
            current_start = None
            current_duration = 0

    # 检查最后一个事件是否需要记录
    if current_start is not None and current_duration >= min_duration:
        events.append({
            'start_index': current_start,
            'end_index': len(koe_idx) - 1,
            'duration': current_duration
        })

    return events


# 筛选KOE-related SST warm events
warm_events = find_koe_events(koe_idx, event_type='warm')
print("Warm events:")
print(warm_events)

# 筛选KOE-related SST cold events
cold_events = find_koe_events(koe_idx, event_type='cold')
print("Cold events:")
print(cold_events)
'''


'''print("检测到的PDO-related SST warm event:")
for idx, event in enumerate(warm_events, 1):
    print(f"事件 {idx}:")
    print(f"  起始索引: {event['start_index']}")
    print(f"  结束索引: {event['end_index']}") 
    print(f"  持续月数: {event['duration']}")
    print(f"  koe_idx范围: {koe_idx[event['start_index']:event['end_index']+1]}")  #+1
    print()'''