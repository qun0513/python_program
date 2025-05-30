import proplot as pplt
import numpy as np
import matplotlib.pyplot as plt
pplt.rc.cycle = '538'
fig, axs = pplt.subplots(figsize=(100,80),dpi=200,ncols=2, span=False, share='labels', refwidth=2.3)
labels = ['a', 'bb', 'ccc', 'dddd', 'eeeee']
hs1, hs2 = [], []

# On-the-fly legends
state = np.random.RandomState(51423)
for i, label in enumerate(labels):
    data = (state.rand(20) - 0.45).cumsum(axis=0)
    h1 = axs[0].plot(
        data, lw=4, label=label, legend='ul',
        legend_kw={'order': 'F', 'title': 'column major'}
    )
    hs1.extend(h1)
    h2 = axs[1].plot(
        data, lw=4, cycle='Set3', label=label, legend='r',
        legend_kw={'lw': 8, 'ncols': 1, 'frame': False, 'title': 'modified\n handles'}
    )
    hs2.extend(h2)

# Outer legends
ax = axs[0]
ax.legend(
    hs1, loc='b', ncols=3, title='row major', order='C',
    facecolor='gray2'
)
ax = axs[1]
ax.legend(hs2, loc='b', ncols=3, center=True, title='centered rows') #center控制框的大小
axs.format(abc="(A)",xlabel='xlabel', ylabel='ylabel', suptitle='A-b-c labels formatting demo')
plt.show()