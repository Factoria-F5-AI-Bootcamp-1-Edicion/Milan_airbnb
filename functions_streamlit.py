import pandas as pd
#La librería folium permite crear mapas interactivos basados en leaflet. Además de la cartografía del mapa,
# permite superponer elementos como puntos y polígonos.
import folium
# Para evitar los espacios en blanco alrededor del mapa se puede utilizar 
# la librería branca
from branca.element import Figure
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab 
import streamlit as st

# Definicion de metodos estaticos

def load_data(path):
    return pd.read_csv(path) 

def load_map(df):
    latitude = df["latitude"].mean()
    longitude = df["longitude"].mean()
    return folium.Map(location=[latitude, longitude], zoom_start=12) 

# Se añaden los puntos de interés (Con CircleMarker añade circulos de tamaño fijo)
# y con MarKer se añaden marcadores
def load_datamap(df, mapa):
    list = range(100)
    print('Longitud de mapa:%d',len(df['id']))
    for i in list:
        print(f'itero {i}')
        latitud = df['latitude'][i]
        longitud = df['longitude'][i]
        titulo = df['host_name'][i]
        folium.Marker([latitud, longitud],tooltip = titulo).add_to(mapa)

# Informacion de las variables
def info_var(tipo,data):
    for col in tipo:
        print(f"Column name: {col}")
        print(data[col].value_counts())
        print()

# ------------------------- TRATAMIENTOS DE DATOS --------------------------------------------------

def prepare_data(data):
    data.drop(["host_id","name","host_name"], axis=1, inplace=True)
    #Obtención de la moda y sustitución de valores perdidos por la moda
    last_review_mode = data["last_review"].mode()[0]
    data["last_review"].fillna(last_review_mode, inplace=True) 
    reviews_por_noche_mean = data["reviews_per_month"].mean()
    data["reviews_per_month"].fillna(reviews_por_noche_mean, inplace=True)
    # Obtener nuevo DataFrame con las entradas cuyo precio es menor a 300 para eliminar outlaiers y
    # teniendo en cuenta que solamente el 3.9 % de las viviendas tiene un precio superior a 300 euros
    return data[(data["price"] < 300)]

def data_cut(data, listLimit, listLabel, column, new_column):
    # Se crea nueva columna con los numeros de reseñas categorizados
    bins=[]
    names= []
    for limit in listLimit:
        bins.append(limit)
    for label in listLabel:
        names.append(label)
    data[new_column] = pd.cut(data[column], bins, labels = names)
    # TODO Ver si se elimina column del dataframe y solo conservar new_column con valores categorizados
    return data

def count_values(column):
    return column.value_counts()

def data_unique(data, column):
    return np.unique(data[column]).tolist()

# ------------------------- FILTROS --------------------------------------------------


def group_by_metrics(aspect,attribute,measure, df_data):
    df_return = pd.DataFrame()
    if(measure == "Absolute"):
        if(attribute == "pass_ratio" or attribute == "tackle_ratio" or attribute == "possession"):
            measure = "Mean"
        else:
            df_return = df_data.groupby([aspect]).sum()            
    
    if(measure == "Mean"):
        df_return = df_data.groupby([aspect]).mean()
        
    if(measure == "Median"):
        df_return = df_data.groupby([aspect]).median()
    
    if(measure == "Minimum"):
        df_return = df_data.groupby([aspect]).min()
    
    if(measure == "Maximum"):
        df_return = df_data.groupby([aspect]).max()
    df_return["aspect"] = df_return.index
    if aspect == "team":
        df_return = df_return.sort_values(by=[attribute], ascending = False)
    return df_return
    
def filter_price(data):
    pass


# ------------------------- GRAFICOS --------------------------------------------------

def show_barplot(data):
    sns.histplot(data=data, x="price", kde=True)

def show_heatmap(data):
    # El método .corr() nos muestra una tabla de correlaciones en el dataset para las variables numericas continuas
    # La diagonal siempre es 1 porque cada variable correlaciona en 1 consigo misma, evidentemente
    corr = data.corr(method = 'spearman')
    # Generamos una máscara para no duplicar lops valores
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Configuramos el matplotlib
    f, ax = plt.subplots(figsize=(11, 9))
    # Ploteamos el heatmap
    sns.heatmap(corr, mask=mask, vmax=1., vmin=-1., center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

def show_catplot(dataf,varx,typed,varhue):
    fig,ax= plt.subplots(ncols=1, figsize=(18,7))
    sns.catplot(data=dataf, x=varx, kind=typed, hue=varhue, ax = ax)
def show_catplot(dataf,varx,typed,varhue):
    fig,ax= plt.subplots(ncols=1, figsize=(18,7))
    sns.catplot(data=dataf, x=varx, kind=typed, hue=varhue, ax = ax)
    st.pyplot(fig)  
# ------------------------- TRANSFORMACIONES --------------------------------------------------

"""ransformar variable numerica para que tenga una distribucion normal
"""
def transform_log(data, column):
    print(f"Distribucion de los valores de la columna{column} antes de la transformacion logaritmica")
    stats.probplot(data[column], dist="norm", plot=pylab)
    pylab.show()
    # Aplicamos logaritmo en base 10 a los valores de la columna
    log = np.log(data[column])
    #   Crea columna con el valor del logaritmo de los valores de column
    data[column] = log
    sns.histplot(x=log, kde=True)
    stats.probplot(data[column], dist="norm", plot=pylab)
    pylab.show()


# Añadir marcadores 
# ==============================================================================
# for row in estaciones_fuel.itertuples():
#     folium.Marker(
#       location = [row.lat, row.lon],
#       popup = row.tag_value,
#    ).add_to(mapa)
# mapa

#folium.Circle(
 #     location = centro_madrid,
  #    radius = 8_000, # metros
   #   popup = '10 km radius',
   #   fill = True,
   #   color = '#B22222'
   #).add_to(mapa)

# Filtrado de elementos por distancia
# ==============================================================================
def price_filter(price, data):
    df = data.cut()
    pass

# Cálculo de distancia al centro de Madrid
#estaciones_fuel['distancias'] = estaciones_fuel.apply(lambda x: price_filter((price, x.lon), centro_madrid),axis = 1)
# Filtrado de las que están a menos de 8 km del centro de Madrid
#estaciones_fuel = estaciones_fuel[estaciones_fuel['distancias'] < 8]
