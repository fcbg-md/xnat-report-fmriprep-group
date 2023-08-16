from bids import BIDSLayout
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
# Imports
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# source data
df = pd.DataFrame({0: {'num': 1, 'label': 'A', 'color': 'red', 'value': 0.4},
                    1: {'num': 2, 'label': 'A', 'color': 'blue', 'value': 0.2},
                    2: {'num': 3, 'label': 'A', 'color': 'green', 'value': 0.3},
                    3: {'num': 4, 'label': 'A', 'color': 'red', 'value': 0.6},
                    4: {'num': 5, 'label': 'A', 'color': 'blue', 'value': 0.7},
                    5: {'num': 6, 'label': 'A', 'color': 'green', 'value': 0.4},
                    6: {'num': 7, 'label': 'B', 'color': 'blue', 'value': 0.2},
                    7: {'num': 8, 'label': 'B', 'color': 'green', 'value': 0.4},
                    8: {'num': 9, 'label': 'B', 'color': 'red', 'value': 0.4},
                    9: {'num': 10, 'label': 'B', 'color': 'green', 'value': 0.2},
                    10: {'num': 11, 'label': 'C', 'color': 'red', 'value': 0.1},
                    11: {'num': 12, 'label': 'C', 'color': 'blue', 'value': 0.3},
                    12: {'num': 13, 'label': 'D', 'color': 'red', 'value': 0.8},
                    13: {'num': 14, 'label': 'D', 'color': 'blue', 'value': 0.4},
                    14: {'num': 15, 'label': 'D', 'color': 'green', 'value': 0.6},
                    15: {'num': 16, 'label': 'D', 'color': 'yellow', 'value': 0.5},
                    16: {'num': 17, 'label': 'E', 'color': 'purple', 'value': 0.68}}
                    ).T

df_input = df.copy()

# split df by labels
labels = df['label'].unique().tolist()
dates = df['num'].unique().tolist()

# dataframe collection grouped by labels
dfs = {}
for label in labels:
    dfs[label]=pd.pivot_table(df[df['label']==label],
                                    values='value',
                                    index=['num'],
                                    columns=['color'],
                                    aggfunc=np.sum)
    
print (dfs)

# find row and column unions
common_cols = []
common_rows = []
for df in dfs.keys():
    common_cols = sorted(list(set().union(common_cols,list(dfs[df]))))
    common_rows = sorted(list(set().union(common_rows,list(dfs[df].index))))

# find dimensionally common dataframe
df_common = pd.DataFrame(np.nan, index=common_rows, columns=common_cols)

# reshape each dfs[df] into common dimensions
dfc={}
for df_item in dfs:
    #print(dfs[unshaped])
    df1 = dfs[df_item].copy()
    s=df_common.combine_first(df1)
    df_reshaped = df1.reindex_like(s)
    dfc[df_item]=df_reshaped

    print (df_reshaped) 

# plotly start 
fig = go.Figure()
# one trace for each column per dataframe: AI and RANDOM
for col in common_cols:
    fig.add_trace(go.Scatter(x=dates,
                             visible=True,
                             marker=dict(size=12, line=dict(width=2)),
                             marker_symbol = 'diamond',name=col
                  )
             )

# menu setup    
updatemenu= []

# buttons for menu 1, names
buttons=[]

# create traces for each color: 
# build argVals for buttons and create buttons
for df in dfc.keys():
    argList = []
    for col in dfc[df]:
        #print(dfc[df][col].values)
        argList.append(dfc[df][col].values)
    argVals = [ {'y':argList}]

    buttons.append(dict(method='update',
                        label=df,
                        visible=True,
                        args=argVals))

# buttons for menu 2, colors
b2_labels = common_cols

# matrix to feed all visible arguments for all traces
# so that they can be shown or hidden by choice
b2_show = [list(b) for b in [e==1 for e in np.eye(len(b2_labels))]]
buttons2=[]
buttons2.append({'method': 'update',
                 'label': 'All',
                 'args': [{'visible': [True]*len(common_cols)}]})

# create buttons to show or hide
for i in range(0, len(b2_labels)):
    buttons2.append(dict(method='update',
                        label=b2_labels[i],
                        args=[{'visible':b2_show[i]}]
                        )
                   )

# add option for button two to hide all
buttons2.append(dict(method='update',
                        label='None',
                        args=[{'visible':[False]*len(common_cols)}]
                        )
                   )

# some adjustments to the updatemenus
updatemenu=[]
your_menu=dict()
updatemenu.append(your_menu)
your_menu2=dict()
updatemenu.append(your_menu2)
updatemenu[1]
updatemenu[0]['buttons']=buttons
updatemenu[0]['direction']='down'
updatemenu[0]['showactive']=True
updatemenu[1]['buttons']=buttons2
updatemenu[1]['y']=0.6

fig.update_layout(showlegend=False, updatemenus=updatemenu)
fig.update_layout(yaxis=dict(range=[0,df_input['value'].max()+0.4]))

# title
fig.update_layout(
    title=dict(
        text= "<i>Filtering with multiple dropdown buttons</i>",
        font={'size':18},
        y=0.9,
        x=0.5,
        xanchor= 'center',
        yanchor= 'top'))

# button annotations
fig.update_layout(
    annotations=[
        dict(text="<i>Label</i>", x=-0.2, xref="paper", y=1.1, yref="paper",
            align="left", showarrow=False, font = dict(size=16, color = 'steelblue')),
        dict(text="<i>Color</i>", x=-0.2, xref="paper", y=0.7, yref="paper",
            align="left", showarrow=False, font = dict(size=16, color = 'steelblue')

                             )
    ])

fig.show()
