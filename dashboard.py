# Importacion de librerias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


PATH_DATA= "milan_airbnb.csv"
def load_data(path):
    return pd.read_csv(path)

# Titulo del dashboard
st.title("Viviendas Airbnb En Milan")

# Cargar archivo csv
df = load_data(PATH_DATA) 

# Para crear barra lateral
st.sidebar.title("Filtrado de viviendas")

# Para correr el streamlit
# streamlit run dashboard.py

alojamientos = df["name"]
latiude = df["latitude"]
longitude = df["longitude"]

st.header("DataSet analizado")

#Mostramos el DataSet
st.write(df)

st.header("Precio por barrio")

#Grafico de precio por barrio
fig1, ax = plt.subplots(ncols=1, figsize=(18,7))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="neighbourhood", y="price", data=df[(df["price"]<300)])
plt.xticks(rotation=90)
st.pyplot(fig1)

#Markdown para gráfico de precio por barrio
st.markdown("La mayoría de precios en todos los barrios rondan los 100€ la noche. En Baggio, Bruzzano, Cantalupa y Quintosole (barrios periféricos de Milán) los precios están por debajo de 50€ la noche. Como ecepción, en Cascina Triulza-Expo, los precios son muy altos")

st.header("Viviendas según tipo de alojamiento, por barrio")

#Grafico de viviendas por barrio y tipo de alojamiento
fig2 = sns.catplot(x ="neighbourhood", hue ="room_type", kind ="count", data = df, aspect=2, height=10)
plt.xticks(rotation=90)
st.pyplot(fig2)

#Markdown para gráfico de tipo de alojamiento por barrio
st.markdown("Entire home/apt (vivienda completa/apartamento), en naranja, predomina como tipo de vivienda más presente en Milán. El segundo tipo de alojamiento más predominante es Private room y también se puede ver que los barrios con más alojamientos son: Buenos Aires - Venezia, Duomo y Navigli, barrios del centro de la ciudad. Los barrios con menos alojamientos se situan a las afueras de la ciudad como por ejemplo en: Cascina Triulza - Expo., mientras que los barrios con más alojamientos son los del centro ")


#Agrupaciones por barrio para los siguientes graficos
ranking_mas_resenas = df.groupby(['neighbourhood'])['number_of_reviews'].agg('sum').sort_values(ascending=False)
df_barrios_review = pd.DataFrame(ranking_mas_resenas)

ranking_mas_dispo = df.groupby(['neighbourhood'])['availability_365'].agg('mean').sort_values(ascending=True)
df_barrios_dispo = pd.DataFrame(ranking_mas_dispo)

df_dispo_review = pd.merge(df_barrios_review, df_barrios_dispo, on='neighbourhood')

df_dispo_review['neighbourhood'] = df_dispo_review.index

st.header("Disponibilidad de los 10 barrios con más reviews")

#Grafico de disponibilidad de los barrios con más reviews
fig3 = sns.catplot(x = "neighbourhood", y = "availability_365", kind = "bar", data =df_dispo_review.head(10))
plt.xticks(rotation=90)
fig3.set( xlabel = "Most rated neighbourhoods (descendent)", ylabel = "availability_365")
st.pyplot(fig3)

#Markdown para gráfico de tipo de disponibilidad de los barrios con más review
st.markdown("Vemos entonces que los barrios con más reviews tienen una disponibilidad media por debajo de los 175 días.")

st.header("Mapa de calor, correlaciones")

#Mapa de calor
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
fig4, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df.corr(), mask=mask, vmax=1., vmin=-1., center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
st.pyplot(fig4)

#Markdown para el mapa de calor
st.markdown("No hay mucha relación entre las variables de este DataSet")

#Correlaciones
#price_cleaned = df.drop(df[(df["price"] < 300)].index)
#stats.probplot(price_cleaned["price"], dist="norm", plot=pylab)

#Categorización de precios y numero de resenas por barrio
bins = [0,300,500,1000,2000,2500]
names = ['0-300','301-500',"501-1000","1001-2000","2001-2500"]
df_cat = df
df_cat["price_cut"] = pd.cut(df["price"], bins, labels = names)

bins = [0,50,200,300,400,700,900]
names = ['0-50','51-200','201-300',"301-400","401-700","701-900"]
df_cat["number_of_reviews_cat"] = pd.cut(df["number_of_reviews"], bins, labels = names)

st.header("Cantidad de viviendas según número de reviews y rangos de precios")

#Canidad de viviendas segun numero de reviews y rango de precios
fig5 = sns.catplot(data=df, x="number_of_reviews_cat", kind="count", hue="price_cut")
st.pyplot(fig5)

#Markdown para cantidad de viviendas segun n de reviews y rango de precios
st.markdown("Las viviendas que tienen mayor numero de reseñas son las viviendas de valor menor a 300 euros (10971) , las cuales tienen una cantidad de reseñas de 1 a 50, aunque hay 5062 viviendas que tienen 0 reseñas")


