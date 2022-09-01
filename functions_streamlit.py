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
import base64
# Definicion de metodos estaticos

def load_data(path):
    return pd.read_csv(path) 

# Informacion de las variables
def info_var(tipo,data):
    for col in tipo:
        print(f"Column name: {col}")
        print(data[col].value_counts())
        print()

#IMAGEN DE FONDO, MAPA DE MILÁN
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

#------------------------AGRUPACIONES --------------------------
def get_barrios(data):
    group = data.groupby(["neighbourhood"])['latitude','longitude','price'].agg('mean')
    data_new = pd.DataFrame(group)
    data_new =data_new[['latitude','longitude','price']]
    return data_new

def get_groupby_number_barrio(data):
    group2 = data.groupby(['neighbourhood']).size().reset_index(name='number')
    data2=pd.DataFrame(group2)
    return data2

# TODO: Implementar estadistica descriptiva basica a variables numericas del dataset
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


# ------------------------------------ MAPA INTERACTIVO DE LIBRERIA FOLIUM -----------------------------

# Carga de mapa de milan con media de la longitud y latitud de todas las viviendas 
def load_map(df):
    latitude = df["latitude"].mean()
    longitude = df["longitude"].mean()
    mapar = folium.Map(location=[latitude, longitude], zoom_start=12) 
    return mapar

# Se añaden los puntos de interés (Con CircleMarker añade circulos de tamaño fijo)
# y con MarKer se añaden marcadores
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
             icono=folium.Icon(color="gray")
        folium.Marker([latitud, longitud],tooltip = titulo, popup=info, icon=icono).add_to(mapa)
    #Leyenda del mapa
    
# Mapa de calor a insertar en el mapa
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

#TODO: Implentar filtro 
def filter_typeroom(data):
    tipo = st.sidebar.radio(
        "¿Tipo de alojamiento?",
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
#Filtro para datos de columnas de variable de tipo numerico en funcion de limites maximo y minimo pasados por parametros   
def filter_per_minmax(data, min,max,column):
    return data[(data[column] >= min) & (data[column] <=max)]


# ------------------------- MENU LATERAL --------------------------------------------------

def menu_lateral(data, tipo, data_map, df_cat):
    if tipo == "Información":
        if st.checkbox('Ver mapa de correlaciones'):
            st.header("Mapa de calor, correlaciones")
            heat_map(data)
            st.markdown("No hay mucha relación entre las variables de este DataSet.")
        st.header("Hipótesis")
        st.components.v1.html(f"""
        <ul>
        <li> ¿El barrio influye en el precio?</li>
        <li> ¿Cuáles son los tipos de alojamientos más alquilados?</li>
        <li> ¿El número de hospedados tiene relación con el tipo de vivienda?</li>
        <li> ¿Cuáles son las fechas de mayor alquiler?</li>
        <li> ¿Las viviendas más caras son las que tienen más reseñas?</li>
        </ul>""", height = 150
         )
        st.header("Conclusiones")
        st.components.v1.html(f"""
        <h3><ol>
    
        <li> Tras quitar los outliers, correspondiente a viviendas céntricas, nos quedamos con viviendas con un rango
        de precios menores a 300 euros. El precio medio de las viviendas es de 85 euros</li>
        <li>Los barrios con más actividad serían los localizados en el centro de la ciudad, los cuales son los de mayor precio.</li>
        <li>El tipo de alojamiento más ofertado es el de los apartamentos completos.</li>
        <li>Al principio teníamos un concepto diferente del "calculated_host_listings_count". Ahora ya sabemos que es el número de alojamientos del mismo dueño</li>
        <li> La calidad de los datos no nos permiten resolver esta hipótesis de las fechas de mayor alquiler<li>
        </ol></h3>""",height = 300
        )
    if tipo == "Mostrar gráficos":
        st.header("Distribucion de precios")
        hist_price(data)
        st.markdown("Se ha decidido a analizar un rango de precios hasta 300 euros por ser donde se concentra la mayor cantidad de viviendas (17454 de las 18322 viviendas totales).")
        st.header("Precio por barrio")     
        show_barrio(data)
        st.markdown("La media de precios ronda los 85 euros por noche. En los barrios Baggio, Bruzzano, Cantalupa y Quintosole (barrios periféricos de Milán) los precios están por debajo de 50€ la noche. Como excepción, en el barrio periférico de  Cascina Triulza-Expo, el precio de las viviendas supera los 200 euros.")
        st.header("Viviendas según tipo de alojamiento, por barrio")
        show_typeroom(data)
         # GRÁFICO NOCHES MÍNIMAS Y TIPO DE ALOJAMIENTO
        st.header("Tipo de alojamiento y mínimo de noches")
        fig, ax = plt.subplots(ncols=1)
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x="room_type", y="minimum_nights", data=data)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        #Markdown para tipo de alojamiento y mínimo de noches
        st.markdown("Las viviendas con mayor reserva de noches se dan en Private room con más de 6 noches, seguido muy de cerca por Entire home/apt con más de 5 noches.")
        st.header("Cantidad de tipos de alojamiento")
        fig, ax = plt.subplots(ncols=1, figsize=(12,7))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=["Entire home/apt", "Private room", "Shared room", "Hotel room"], y=[13605, 4376, 267, 74], data=data)
        st.pyplot(fig)
        st.markdown("Amplia mayoría de casas y apartamentos completos. El tipo Hotel room tiene una presencia prácticamente testimonial")

        st.markdown("Entire home/apt (vivienda completa/apartamento), en naranja, predomina como tipo de vivienda más presente en Milán. El segundo tipo de alojamiento más predominante es Private room y también se puede ver que los barrios con más alojamientos son: Buenos Aires - Venezia, Duomo y Navigli, barrios del centro de la ciudad. Los barrios con menos alojamientos se situan a las afueras de la ciudad como por ejemplo en: Cascina Triulza - Expo., mientras que los barrios con más alojamientos son los del centro.")
        st.header("Disponibilidad de los 10 barrios con más reviews")
        show_barriosreviews(df_cat)
        st.markdown("Vemos entonces que los barrios con más reviews tienen una disponibilidad media por debajo de los 175 días.")
        st.header("Cantidad de viviendas según número de reviews y rangos de precios.")
        show_reviewsprice(df_cat)
        st.markdown("Las viviendas que tienen mayor numero de reseñas son las viviendas de valor menor a 300 euros (10971) , las cuales tienen una cantidad de reseñas de 1 a 50, aunque hay 5062 viviendas que tienen 0 reseñas.")  
       

    if tipo == "Mostrar mapa":
       #Cargar mapa
        mapa = load_map(data_map)
        load_datamap(data_map,mapa)
        heatmap_map(mapa,data_map)

        #fs.load_datamap(data_map,mapa)
        # Renderizar mapa
        st.header("Mapa interactivo de calor por barrios")
        st_folium(mapa, width=800)
        st.components.v1.html(f"""
        <p>Aquí se muestra un mapa de calor con distribución de viviendas , clasificado por barrios
        Los puntos de interes de <font color ="red">color rojo</font> indican barrios con mas de 400 viviendas ofertadas
        Los puntos de interes de <font color ="orange">color naranja</font> indican barrios que tienen entre 200 y 400 viviendas ofertadas
        Los puntos de interes de <font color ="gray">color gris</font> indican barrios con menos de 200 viviendas ofertadas
        </p>
        """,height = 300
        )


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
    fig, ax = plt.subplots(ncols=1, figsize=(18,10))
    fig= sns.catplot(x ="neighbourhood", hue ="room_type", kind ="count", data = data, aspect=2, height=10)
    ax=sns.set_theme(style="whitegrid")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def hist_price(dataf):
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
    ax=sns.set_theme(style="whitegrid")
    ax= sns.histplot(data=dataf, x="price", kde=True)
    plt.xticks(rotation=90)
    st.pyplot(fig)

def show_reviewsprice(data):
    group = data.groupby(['number_of_reviews_cat', 'price_cut'])
    data_prices = pd.DataFrame(group.size().unstack())
    st.table(data_prices)
    fig = sns.catplot(data=data, x="number_of_reviews_cat", kind="count", hue="price_cut")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def show_barriosreviews(df_dispo_review):
    fig3 = sns.catplot(x = "neighbourhood", y = "availability_365", kind = "bar", data =df_dispo_review.head(10))
    plt.xticks(rotation=90)
    fig3.set( xlabel = "Most rated neighbourhoods (descendent)", ylabel = "availability_365")
    st.pyplot(fig3)

# TODO: CORREGIR ERROR AL CARGAR VISUALIZACION
def heat_map(data):
    mask = np.triu(np.ones_like(data.corr(method= 'kendall'), dtype=bool))
    fig4, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(data.corr(method= 'kendall'), mask=mask, vmax=1., vmin=-1., center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    st.pyplot(fig4)

def price_dist(data):
    f =stats.probplot(data["price"], dist="norm", plot=pylab)
    fig = plt.figure()
    fig.add_subplot(f)
    st.pyplot(fig)

def show_barrio(dataf):
    fig, ax = plt.subplots(ncols=1, figsize=(18,7))
    # filtro =  st.radio(
    #  "Ingrese rango de precios a filtrar",
    #  ("0-25","25-50", "50-75","75-100","100-150","150-300"))
    # if filtro == "0-25":
    #     dataf = filter_per_minmax(dataf,0,25,'price')
    # elif filtro == "25-50":
    #     dataf =filter_per_minmax(dataf,26,50,'price')
    # elif filtro == "50-75":
    #     dataf =filter_per_minmax(dataf,51,75,'price')
    # elif filtro == "75-100":
    #     dataf =filter_per_minmax(dataf,76,100,'price')
    # elif filtro == "100-150":
    #     dataf =filter_per_minmax(dataf,101,150,'price')
    # elif filtro == "150-300":
    #     dataf =filter_per_minmax(dataf,151,300,'price')
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