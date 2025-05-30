from multiprocessing import Pool
import time
import numpy as np

def f(mdl):
    koe_idx=np.loadtxt(f"D:/decadal prediction/results/piControl/{mdl}/koeindex.txt")
    print(koe_idx)
    print(mdl)



if __name__ == '__main__':
    with Pool(6) as p:
        # 如果您需要收集返回值，可以使用 p.starmap 并提供一个包含参数和函数的元组列表
        # p.starmap(f, [(i,) for i in np.arange(0, 3)])
        p.map(f, ['BCC-CSM2-MR','CanESM2','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorCPM1'])


