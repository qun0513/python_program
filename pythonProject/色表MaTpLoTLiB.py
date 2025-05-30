import matplotlib as plm
import matplotlib.pyplot as plt

colors = plm.colors.cnames.keys()

fig = plt.figure('百里希文', facecolor='lightyellow', edgecolor='k')
axes = fig.subplots(len(colors) // 4, 4)

for c, ax in zip(colors, axes.ravel()):
    ax.hist(1, 3, color=c)
    ax.text(1.2, 0.1, c)
    ax.set_axis_off()
fig.subplots_adjust(left=0, bottom=0, right=0.9,
                    top=1, hspace=0.1, wspace=0.1)

plt.show()