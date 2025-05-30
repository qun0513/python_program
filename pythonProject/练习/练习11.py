import matplotlib.pyplot as plt
import numpy as np
fig,ax=plt.subplots()
counts=[20,15,12,16]
weather=['drizzle','fog','rain','sun']
ax.pie(counts,labels=weather,autopct='%1.1f%%',textprops={'fontsize':16})
plt.show()