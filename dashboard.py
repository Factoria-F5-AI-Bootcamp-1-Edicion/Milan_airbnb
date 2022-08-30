# Importacion de librerias
from sqlite3 import DatabaseError
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import functions_streamlit as fs
from streamlit_folium import st_folium
import pylab 
import scipy.stats as stats


#Ruta del archivo contenedor del DATASET
PATH_DATA= "milan_airbnb.csv"

# Cargar archivo csv
df = fs.load_data(PATH_DATA) 

data_map = df
#Cargar mapa
mapa = fs.load_map(data_map)
fs.load_datamap(data_map,mapa)
# Renderizar mapa
st_folium(mapa, width=625)
# Preparo los datos a utilizar en funcion de la exploracion de datos basica hecha previamente
df = fs.prepare_data(df)
# Clasificamos las columnas segun el tipo de variable (numerica o categorica)
numericas = ["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
categoricas =["neighbourhood","room_type","last_review", "latitude", "longitude"]

# Categorizo la variable precio
df_cat = fs.data_cut(df,[0,50,100,150,200,250,300], ['0-50','51-100','101-150',"151-200","201-250","251-300"],
 'price', 'price_cut')

# Categorizo la variable number_of_reviews
df_cat = fs.data_cut(df_cat,[-1,0,50,200,300,400,700,900],['0 reviews','1-50','51-200','201-300',"301-400","401-700","701-900"],
"number_of_reviews","number_of_reviews_cat")

# Datos del menu
precios = {"type":"price", "data": fs.data_unique(df, "price")}
barrios = {"type":"neighbourhood", "data":fs.data_unique(df, "neighbourhood")}

# Titulo del dashboard
st.title("Panel de visualizacion")

if st.checkbox("Mostrar DataSet utilizado : "):
    st.dataframe(data_map)

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
def show_barrio(dataf):
    st.subheader("Barrio discriminado por tipo de alojamiento")
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="neighbourhood", y="price", data=dataf)
    plt.xticks(rotation=90)
    st.pyplot(fig)

def graph_menu(data):
    data_ret = data
    show_barrio(data)
    data_ret = filter_typeroom(data)
    return data_ret
  

def menu_lateral(data, tipo):
    st.sidebar.title("Filtros")
    if tipo == "Mostrar graficos":     
        graph_menu(data)
    if tipo_visualizacion == "Mostrar mapa":
        pass


# RADIO ELECCIÓN TIPO VIVIENDA
tipo_visualizacion = st.sidebar.radio(
     "¿Qué tipo de visualizacion quieres?",
     ("Mostrar mapa", "Mostrar graficos"))

def show_typeroom(data):
    st.subheader("Precio por tipo de vivienda")
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
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
    group = df.groupby(['number_of_reviews_cat', 'price_cut'])
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

def price_dist(data):
    f =stats.probplot(data["price"], dist="norm", plot=pylab)
    fig = plt.figure()
    fig.add_subplot(f)
    st.pyplot(fig)

menu_lateral(df,tipo_visualizacion)
show_typeroom(df)
hist_price(df)
show_barriosreviews(df_cat)
show_reviewsprice(df_cat)
heat_map(df)
price_dist(df)









