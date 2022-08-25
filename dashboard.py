# Importacion de librerias
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


#Ruta del archivo contenedor del DATASET
PATH_DATA= "milan_airbnb.csv"

# Cargar archivo csv
df = fs.load_data(PATH_DATA) 

# Clasificamos las columnas segun el tipo de variable (numerica o categorica)
numericas = ["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
categoricas =["neighbourhood","room_type","last_review", "latitude", "longitude"]
# Titulo del dashboard
st.title("Panel de visualizacion")

#Cargar mapa
mapa = fs.load_map(df)

#Cargar datos de viviendas en mapa
fs.load_datamap(df,mapa)

# Renderizar mapa
st_folium(mapa, width=625)

# Preparo los datos a utilizar en funcion de la exploracion de datos basica hecha previamente
df = fs.prepare_data(df)


# Categorizo la variable precio
df_cat = fs.data_cut(df,[0,50,100,150,200,250,300], ['0-50','51-100','101-150',"151-200","201-250","251-300"],
 'price', 'price_cut')


# Categorizo la variable number_of_reviews
df_cat = fs.data_cut(df_cat,[0,50,200,300,400,700,900],['0-50','51-200','201-300',"301-400","401-700","701-900"],
"number_of_reviews","number_of_reviews_cat")
# st.dataframe(df_cat)

# Lista de variables de entrada
numericas = ["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
categoricas =["neighbourhood","room_type","last_review"]
#muestra dataframe si selecciono enlace

# TODO: Corregir errores en visualizacion de graficos

# Mostrar catplot de rangos de numeros de reseñas comparado con rango de precios
fs.show_catplot(df_cat,'number_of_reviews_cat','count','price_cut')

fig,ax= plt.subplots(ncols=1, figsize=(18,7))
sns.catplot(x ="neighbourhood", hue ="room_type", kind ="count", data = df, aspect=2, height=10)
plt.xticks(rotation=90)
st.pyplot(fig)

if st.button("Mostrar"):
    st.dataframe(df)
if st.button("Ocultar"):
    st.write("")

st.sidebar.title("Filtros")
# Mostrar por barrio
# Obtengo los elementos unicos de la columna room_type
list = np.unique(df["room_type"])

if st.sidebar.checkbox("Mostrar precio por tipo de Barrio", False, key=1):
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
    sns.boxplot(y='price', x='neighbourhood', data=df, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)  
    
select = st.sidebar.selectbox('Seleccione un barrio',list )
rt_data = df[df["room_type"]== select]





# Gráfico para comparar los precios de los distintos barrios


sns.histplot(data=df[df["price"]< 200], x="price", kde=True)
