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

# Para crear barra lateral
st.sidebar.title("Barra lateral")

# Para correr el streamlit
# streamlit run dashboard.py


