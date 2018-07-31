import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import plotly.plotly as py
import pandas as pd

# kaggle dataset
housing_data = pd.read_csv('dataset/housing/housing.csv')

df_x = housing_data.iloc[:, 2:10]

df = pd.read_csv('dataset/2014_us_cities.csv')
df.head()

df['text'] = df['name'] + '<br>Population ' + (df['pop'] / 1e6).astype(str) + ' million'
limits = [(0, 2), (3, 10), (11, 20), (21, 50), (50, 3000)]
colors = ["rgb(0,116,217)", "rgb(255,65,54)", "rgb(133,20,75)", "rgb(255,133,27)", "lightgrey"]
cities = []
scale = 5000

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    city = dict(
        type='scattergeo',
        locationmode='USA-states',
        lon=df_sub['lon'],
        lat=df_sub['lat'],
        text=df_sub['text'],
        marker=dict(
            size=df_sub['pop'] / scale,
            color=colors[i],
            line=dict(width=0.5, color='rgb(40,40,40)'),
            sizemode='area'
        ),
        name='{0} - {1}'.format(lim[0], lim[1]))
    cities.append(city)

layout = dict(
    title='2014 US city populations<br>(Click legend to toggle traces)',
    showlegend=True,
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showland=True,
        landcolor='rgb(217, 217, 217)',
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)"
    ),
)

fig = dict(data=cities, layout=layout)
py.iplot(fig, validate=False, filename='d3-bubble-map-populations')
