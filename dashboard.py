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

st.image('sidebar.png')
# Titulo del dashboard
st.title("Alojamientos Airbnb En Milan")
page = """
<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<head>
<body>
<hr>
<h4>Autores</h4>
<ul class="list-group">
  <li class="list-group-item">Nayare</li>
  <li class="list-group-item">Anghie</li>
  <li class="list-group-item">Javier</li>
  <li class="list-group-item">Sebastian</li>

</ul>
<hr>
<br>
</body>
"""
st.markdown(page, unsafe_allow_html=True)


#Descripción de lo analizado
st.write("A continuación mostramos algunos gráficos y conclusiones que hemos podido extraer, con los datos proporcionados, sobre el alquiler de alojamientos en Milán a través de la APP AirBnB. El objetivo es encontrar patrones relevantes de cara al alquiler de viviendas con fines vacacionales.")

# Cargar archivo csv
df = fs.load_data(PATH_DATA) 
df = df[df["price"] <300]
data_map = df.copy()


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


if st.checkbox("Selecciona para mostrar DataSet utilizado : "):

    st.header("DataSet analizado")
    st.write("Las variables que encontramos en el Dataset son:")
    st.components.v1.html(f"""<p>Estas son las variables por entrada que contiene este dataset:  <ul>
    <li><b>id =</b>Identificación de la vivienda </li>
    <li><b>name =</b>Nombre de la vivienda y descripción </li>
     <li><b>host_id =</b>id del anfitrión </li>
      <li><b>host_name =</b>Nombre del anfitrión </li>
    <li><b>neighbourhood =</b>Nombre del barrio </li>
    <li><b>Latitude =</b>Cordenada de latitud de la vivienda </li>
     <li><b>Longitude =</b>Cordenada de longitud de la vivienda </li>
     <li><b>Room_type =</b>Tipo de vivienda </li>
     <li><b>Price =</b>Precio por noche de la vivienda </li>
     <li><b>minimum_nights =</b>Estadía mínima de la vivienda  </li>
    <li><b>number_of_reviews=</b>Número de reseñas  </li>
     <li><b>last_of_reviews=</b>Fecha de última reseña </li>
    <li><b>reviews_per_month=</b> Reseñas por mes</li>
    <li><b>calculated_host_listings_count=</b>Número de viviendas del mismo dueño.</li>
    <li><b>availability_365=</b> Días disponibles al año</li>
    </ul></p>""",scrolling=True)
    
    st.dataframe(df_cat.drop(['latitude','longitude'],axis=1))
    st.markdown("Se eliminaron las columnas de host_id, host_name y name por considerarla que no son variables que influyan en nuestra salida")



# RADIO ELECCIÓN TIPO VIVIENDA
tipo_visualizacion = st.sidebar.radio(
     "Tipo de visualizacion",
     ("Informacion","Mostrar mapa", "Mostrar graficos"))

st.sidebar.image('sidebar.png')

fs.menu_lateral(df,tipo_visualizacion, data_map, df_cat)











