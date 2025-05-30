'''
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from vega_datasets import data
#data = pd.DataFrame({'x':list('ABCDE') , 'y':[5,3,6,7,2]})
#print(data)
pd_iris=data.iris
alt.Chart(pd_iris).mark_bar().encode(x='x').interactive()
#mark_(point,line,bar,circle,boxplot,geoshpae,text,
#      area,rule,square,rect,tick)
#data mark encode
plt.interactive(True)
plt.show()
'''
#动图制作

import plotly.express as px
from vega_datasets import data
a=data.list_datasets()
print(a)
print(1,2,3,4)
#print(data.anscombe)
b=data.iris()
print(b)
'''
df = data.disasters()
df = df[df.Year > 1990]
fig = px.bar(df,
             y="Entity",
             x="Deaths",
             animation_frame="Year",
             orientation='h',
             range_x=[0, df.Deaths.max()],
             color="Entity")
# improve aesthetics (size, grids etc.)
fig.update_layout(width=1000,
                  height=800,
                  xaxis_showgrid=False,
                  yaxis_showgrid=False,
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  title_text='Evolution of Natural Disasters',
                  showlegend=False)
fig.update_xaxes(title_text='Number of Deaths')
fig.update_yaxes(title_text='')
fig.show()
'''

