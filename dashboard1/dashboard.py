# Importacion de librerias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

PATH_DATA= "milan_airbnb.csv"
def load_data(path):
    return pd.read_csv(path)

# Titulo del dashboard
st.title("Panel de visualizacion")

# Cargar archivo csv
df = load_data(PATH_DATA) 
df.drop(["host_id","name","host_name","latitude","longitude"], axis=1, inplace=True)
# Lista de variables de entrada
numericas = ["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
categoricas =["neighbourhood","room_type","last_review"]
#muestra dataframe si selecciono enlace
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
