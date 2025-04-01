import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import requests
import nltk
import re
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import io
from  utils.text_processing import clean_text, get_ngrams, get_text_stats, find_pattern, download_nltk_resources

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis de Sentimientos en Reseñas de Steam",
    page_icon="🎮",
    layout="wide"
)

# Título principal
st.title("Análisis de Sentimientos en Reseñas de Steam")
st.markdown("Este es un proyecto para practicar el procesamiento de lenguaje natural con reseñas de videojuegos de la API de Steam.")

# Asegurar que los recursos de NLTK estén descargados
download_nltk_resources()

# Sección de configuración
st.header("Configuración")

col1, col2, col3 = st.columns(3)
with col1:
    id_game = st.text_input("ID del juego", "2246340", help="Por ejemplo: Monster Hunter Wilds = 2246340")
with col2:
    language = st.selectbox("Idioma de las reseñas", ["spanish", "english", "french", "german"], index=0)
with col3:
    num_per_page = st.slider("Número de reseñas", min_value=10, max_value=100, value=100, step=10)
