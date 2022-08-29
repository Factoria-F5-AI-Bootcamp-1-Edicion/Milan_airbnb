# Importacion de librerias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# RUTA DEL ARCHIVO CSV, LO ALMACENAMOS EN UNA VARIABLE
PATH_DATA= "milan_airbnb.csv"
def load_data(path):
    return pd.read_csv(path)

#IMAGEN DE FONDO, MAPA DE MILÁN
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('plano_milan_02.png')     


# TÍTULO DEL DASHBOARD
st.title("Airbnb en Milán")


# IMPORTAR DATASET
df = load_data(PATH_DATA) 

# MAPA CON ÁREA DE INFLUENCIA
st.map(data=None, zoom=None, use_container_width=True)

# PARA CREAR UNA BARRA LATERAL
st.sidebar.title("FILTROS")   

# FILTRO PARA SELECCIONAR CANTIDAD DE DINERO A INVERTIR
bins = [0,300,500,1000,2000,2500]
rango_pasta = ["0-300","301-500","501-1000","1001-2000","2001-2500"]
df_cat = df
df_cat["price_cut"] = pd.cut(df["price"], bins, labels = rango_pasta)
df_cat["price_cut"].value_counts()

add_selectbox = st.sidebar.selectbox(
    "¿En qué rango económico te mueves?",
     (rango_pasta[0],rango_pasta[1],rango_pasta[2],rango_pasta[3],rango_pasta[4])
)

# FILTRO PARA SELECCIONAR TIPO DE ALOJAMIENTO
tipos_alojamiento = ["Entire home/apt","Private room","Shared room","Hotel room"]
add_selectbox = st.sidebar.selectbox(
    "¿Qué tipo de alojamiento quieres?",
      (tipos_alojamiento[0],tipos_alojamiento[1],tipos_alojamiento[2],tipos_alojamiento[3])
)
# FILTRO PARA SELECCIONAR BARRIO
# df_barrios = df.index_col("neighbourhood")
# df_barrios = df.set_index("neighbourhood")
df_barrios = df.drop_duplicates(subset=["neighbourhood"]) 
add_selectbox = st.sidebar.selectbox(
    "¿En qué barrio quieres pernoctar?",
     (df_barrios["neighbourhood"])
)
# RADIO ELECCIÓN TIPO VIVIENDA
tipo = st.sidebar.radio(
     "¿Qué tipo de alojamiento quieres?",
     ("Entire home/apt", "Private room", "Shared room ", "Hotel room"))

if tipo == "Entire home/apt":
     st.write("Has elegido Entire home/apt")
if tipo == "Private room":
     st.write("Has elegido Private room")
if tipo == "Shared room":
     st.write("Has elegido Shared room")     
if tipo == "Hotel room":
     st.write("Has elegido Hotel room")  

# SUBTÍTULO
st.header("Gráficos relevantes")

#GRÁFICOS






# Para correr el streamlit
# streamlit run dashboard.py






