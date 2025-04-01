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
    num_per_page = st.slider("Número de reseñas", min_value=10, max_value=100, value=100, step=1)

# Botón para obtener las reseñas
if st.button("Obtener Reseñas"):
    with st.spinner("Descargando reseñas de Steam..."):
        # URL de la API de Steam para obtener reseñas
        url = f"https://store.steampowered.com/appreviews/{id_game}?json=1&language={language}&num_per_page={num_per_page}"

        try:
            # Realizar la solicitud a la API
            response = requests.get(url)
            data = response.json()

            # Extraer las reseñas del JSON
            reviews = data.get('reviews', [])

            if not reviews:
                st.error("No se encontraron reseñas para este juego con los parámetros especificados.")
                st.stop()

            # Crear un DataFrame a partir de las reseñas
            df_reviews = pd.DataFrame(reviews)

            # Guardar el DataFrame en la sesión
            st.session_state.df_reviews = df_reviews
            st.session_state.game_id = id_game

            st.success(f"Se han descargado {len(reviews)} reseñas correctamente!")
        except Exception as e:
            st.error(f"Error al obtener las reseñas: {str(e)}")
            st.stop()

# Verificar si el DataFrame esta en la sesion
if 'df_reviews' not in st.session_state:
    st.info("Por favor, obten las reseñas primero usando el boton de arriba.")
    st.stop()

df_reviews = st.session_state.df_reviews

# Mostrar informacion basica del dataframe
st.header("Informacion del Dataset")
st.write(f"Numero de reseñas: {len(df_reviews)}")

# Mostrar las primeras filas del Dataframe
with st.expander("Ver primeras filas del DataFrame"):
    st.write("Columnas principales del Dataframe:")
    if 'review' in df_reviews.columns:
        simple_df = df_reviews[['review', 'voted_up', 'author', 'timestamp_created']]
        st.dataframe(simple_df.head())
    else:
        st.warning("El DataFrame no contiene la columna 'review' esperada.")