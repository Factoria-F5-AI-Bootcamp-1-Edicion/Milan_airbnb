from turtle import width
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
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import branca.colormap as cm
# Definicion de metodos estaticos

def load_data(path):
    return pd.read_csv(path) 

def load_map(df):
    latitude = df["latitude"].mean()
    longitude = df["longitude"].mean()
    mapar = folium.Map(location=[latitude, longitude], zoom_start=12) 
    return mapar

# Se añaden los puntos de interés (Con CircleMarker añade circulos de tamaño fijo)
# y con MarKer se añaden marcadores
def get_barrios(data):
    group = data.groupby(["neighbourhood"])['latitude','longitude','price'].agg('mean')
    data_new = pd.DataFrame(group)
    data_new =data_new[['latitude','longitude','price']]
    return data_new

def get_groupby_number_barrio(data):
    group2 = data.groupby(['neighbourhood']).size().reset_index(name='number')
    data2=pd.DataFrame(group2)
    return data2

def format(*args):
    value = 0
    for n in args:
        value += str(n)
        value += '\n'
    return value

def load_datamap(df, mapa):
    datalat_long= get_barrios(df)
    datan = get_groupby_number_barrio(df)
    list = df.sort_values('neighbourhood').neighbourhood.unique().tolist()
    length= range(len(datalat_long))
    for i in length:
        latitud = datalat_long.iloc[i]['latitude']
        longitud = datalat_long.iloc[i]['longitude']
        titulo = "Barrio: "+str(list[i])+ " Viviendas:" +"\n"+ str(datan.iloc[i]['number']) 
        info = titulo + "       Precio Promedio : "+str(round(datalat_long.iloc[i]['price'],2))+" euros"
        if int(datan.iloc[i]['number']) > 400:
            icono=folium.Icon(color="red", icon="info-sign")
        elif int(datan.iloc[i]['number']) > 200 :
            icono=folium.Icon(color="orange")
        else: 
             icono=folium.Icon(color="white")
        folium.Marker([latitud, longitud],tooltip = titulo, popup=info, icon=icono).add_to(mapa)
    #Leyenda del mapa
    

# Informacion de las variables
def info_var(tipo,data):
    for col in tipo:
        print(f"Column name: {col}")
        print(data[col].value_counts())
        print()
# -------------------------MAPAS ---------------------------------------------
def heatmap_map(mapa,data):
    fig = Figure(width=600, height=400)
    fig.add_child(mapa)
    heatmap = HeatMap(
            data = data[['latitude','longitude']].to_numpy(),
            radius = 13.5,
            blur = 25,   
        )
    heatmap.add_to(mapa)
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

def filter_typeroom(data):
    tipo = st.sidebar.radio(
        "¿Qué tipo de alojamiento quieres?",
        ("Ninguno","Entire home/apt", "Private room", "Shared room", "Hotel room"))
    if tipo == "Entire home/apt":
        return data[(data["room_type"] == "Entire home/apt" )]
    if tipo == "Private room":
        return data[(data["room_type"] == "Private room" )]
    if tipo == "Shared room":
        return data[(data["room_type"] == "Shared room" )]  
    if tipo == "Hotel room":
        return data[(data["room_type"] == "Hotel room" )]
    if tipo == "Ninguno":
        return data

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

# ------------------------- MENU LATERAL --------------------------------------------------

def menu_lateral(data, tipo, data_map, df_cat):

    if tipo == "Mostrar graficos":     
        show_barrio(data)
        hist_price(data)
        show_barriosreviews(df_cat)
        show_reviewsprice(df_cat)
        heat_map(data)
        price_dist(data)
        
    if tipo == "Mostrar mapa":
       #Cargar mapa
        mapa = load_map(data_map)
        load_datamap(data_map,mapa)
        heatmap_map(mapa,data_map)
        #fs.load_datamap(data_map,mapa)
        # Renderizar mapa
        st_folium(mapa, width=800)
# ------------------------- GRAFICOS --------------------------------------------------

def show_priceneig(df):
    #Titulo del grafico
    st.header("Precio por barrio")
    #Grafico de precio por barrio
    fig1, ax = plt.subplots(ncols=1, figsize=(18,7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="neighbourhood", y="price", data=df[(df["price"]<300)])
    plt.xticks(rotation=90)
    st.pyplot(fig1)


def show_typeroom(data):
    st.subheader("Precio por tipo de vivienda")
    fig, ax = plt.subplots(ncols=1, figsize=(18,10))
    fig= sns.catplot(x ="neighbourhood", hue ="room_type", kind ="count", data = data, aspect=2, height=10)
    ax=sns.set_theme(style="whitegrid")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def hist_price(dataf):
    st.subheader("Distribucion de precios")
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
    ax=sns.set_theme(style="whitegrid")
    ax= sns.histplot(data=dataf, x="price", kde=True)
    plt.xticks(rotation=90)
    st.pyplot(fig)

def show_reviewsprice(data):
    st.subheader("Rango de cantidad de reseñas (filas) agrupadas por precio (columnas)")
    group = data.groupby(['number_of_reviews_cat', 'price_cut'])
    data_prices = pd.DataFrame(group.size().unstack())
    st.table(data_prices)
    fig = sns.catplot(data=data, x="number_of_reviews_cat", kind="count", hue="price_cut")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def show_barriosreviews(df_dispo_review):
    st.subheader("Disponibilidad de los 10 barrios con mas reseñas")
    fig = sns.catplot(x = "neighbourhood", y = "availability_365", kind = "bar", data=df_dispo_review.head(17))
    plt.xticks(rotation=90)
    fig.set( xlabel = "Most rated neighbourhoods (descendent)", ylabel = "availability_365")
    st.pyplot(fig)
# TODO: CORREGIR ERROR AL CARGAR VISUALIZACION
def heat_map(data):
    corr = data.corr('spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    f = sns.heatmap(corr, mask=mask, vmax=1., vmin=-1., center=0, 
    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax = ax)
    st.write(f)
# TODO: CORREGIR ERROR AL CARGAR VISUALIZACION
def price_dist(data):
    f =stats.probplot(data["price"], dist="norm", plot=pylab)
    fig = plt.figure()
    fig.add_subplot(f)
    st.pyplot(fig)

def show_barrio(dataf):
    st.subheader("Barrio discriminado por tipo de alojamiento")
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="neighbourhood", y="price", data=dataf)
    plt.xticks(rotation=90)
    st.pyplot(fig)

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
