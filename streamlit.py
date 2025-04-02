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
nltk.download('punkt')
nltk.download('stopwords')
#download_nltk_resources()

# Sección de configuración
st.header("Configuración")
st.markdown("*Para sacar el **ID del juego** ve a la tienda de [STEAM](https://store.steampowered.com/) y busca un juego y obten el id del juego*")

col1, col2, col3 = st.columns(3)
with col1:
    id_game = st.text_input("ID del juego", "2246340", help="Por ejemplo: Monster Hunter Wilds = 2246340")
with col2:
    language = st.selectbox("Idioma de las reseñas", ["spanish", "english", "french", "german"], index=0)
with col3:
    num_per_page = st.slider("Número de reseñas", min_value=10, max_value=100, value=100, step=1)

#==============================================================================================================
#==================================== Seccion 1: Obtener Reseñas ==============================================
#==============================================================================================================

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

#==============================================================================================================
#==================================== Seccion 2: Analisis de sentimiento ======================================
#==============================================================================================================
# Seccion de analisis de sentimiento
st.header("Analisis de sentimiento")

# Creando columna de  'sentiment' basandonos de la columna de 'voted_up'
if 'voted_up' in df_reviews.columns:
    df_reviews['sentiment'] = df_reviews['voted_up'].apply(lambda x: 'Positive' if x else 'Negative')

    # contar la cantidad de reseñas por sentimiento
    sentiment_counts = df_reviews['sentiment'].value_counts()

    #Graficar el resultado (pastel)
    fig, ax = plt.subplots(figsize = (3, 3))
    ax.pie(sentiment_counts, labels= sentiment_counts.index, autopct='%1.1f', startangle=90)
    ax.axis('equal')
    ax.set_title(f'Porcentaje de reseñas del juego {st.session_state.game_id} en steam')
    st.pyplot(fig)

    #Mostrar conteo exacto
    st.write('Conteo de Sentimientos:')
    st.write(pd.DataFrame(sentiment_counts).transpose())
else:
    st.warning("No se encontro la columna 'voted_up' en los datos.")

#==============================================================================================================
#==================================== Seccion 3: Reseñas Largas / Cortas ======================================
#==============================================================================================================

# Seccion de reseñas largas y cortas
st.header("Reseñas largas y cortas")

if 'review' in df_reviews.columns and len(df_reviews) > 0:
    #Calcular longitud de cada reseña
    df_reviews['review_length'] = df_reviews['review'].apply(len)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Reseña mas larga')
        longest_review = df_reviews.loc[df_reviews['review_length'].idxmax()]['review']
        st.text_area("", longest_review, height=200, disabled=True)

    with col2:
        st.subheader('Reseña mas corta')
        min_length_review = df_reviews.loc[df_reviews['review_length'].idxmin()]['review']
        st.text_area("", min_length_review, height=200, disabled=True)

    # Reseña aleatoria
    st.subheader('Reseña aleatoria')
    if st.button("Selecciona Reseña Aleatria"):
        random_index = random.randint(0, len(df_reviews) - 1)
        st.session_state.random_review = df_reviews['review'][random_index]
        st.session_state.random_index = random_index

    if 'random_review' not in st.session_state:
        random_index = random.randint(0, len(df_reviews)-1)
        st.session_state.random_review = df_reviews['review'][random_index]
        st.session_state.random_index = random_index

    st.text_area("Reseña original", st.session_state.random_review, height=150, disabled=True)

#==================================== Uso de Stopwords ======================================
    # Tokenizar y limpiar la reseña
    clean_words = clean_text(st.session_state.random_review, language=language)
    clean_review = " ".join(clean_words)

    st.text_area("Reseña limpia (ya aplicado stopwords)", clean_review, height=150, disabled=True)

#==============================================================================================================
#==================================== Seccion 4: Estadisticas de texto ========================================
#==============================================================================================================

#Estadisticas de texto
st.header("Numero de palabras")

if 'review' in df_reviews.columns:
    #juntar todas las reseñas en un solo texto
    all_reviews = " ".join(df_reviews['review'].tolist())

    #Obtener estadisticas de texto
    text_stats = get_text_stats(all_reviews, language=language)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de palabras", text_stats['word_count'])
    with col2:
        st.metric("Palabras unicas", text_stats['unique_count'])
    with col3:
        st.metric("Promedio de palabras por reseñas", round(text_stats['word_count'] / len(df_reviews), 2))

#==============================================================================================================
#==================================== Seccion 5: Palabras comunes =============================================
#==============================================================================================================

#Palabras comunes
st.header("Palabras Comunes")

if 'review' in df_reviews.columns:
    # Usar estadisticas previamente calculadas
    word_freq = Counter(text_stats['clean_tokens'])
    most_common_words = word_freq.most_common(10)
    words, counts = zip(*most_common_words)

    #graficar las palabras mas frecuentes
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts), ax=ax)
    ax.set_title("Palabras mas repetidas en las reseñas")
    ax.set_xlabel("Palabra")
    ax.set_ylabel("Frecuencia")
    plt.xticks(rotation=45)
    st.pyplot(fig)

#==============================================================================================================
#==================================== Seccion 6: Palabras Unicas ==============================================
#==============================================================================================================
# Sección de palabras únicas
st.header("Palabras Únicas")

if 'review' in df_reviews.columns:
    # Obtener palabras que aparecen solo una vez
    rare_words = [word for word, count in word_freq.items() if count == 1]

    if len(rare_words) > 20:
        sample_rare = random.sample(rare_words, 20)
        st.write(f"Mostrando de 20 palabras únicas (de un total de {len(rare_words)}):")
        st.write(", ".join(sample_rare))
    else:
        st.write(f"Palabras que aparecen solo una vez ({len(rare_words)}):")
        st.write(", ".join(rare_words))

#==============================================================================================================
#==================================== Seccion 7: WordCloud ====================================================
#==============================================================================================================

# Sección de WordCloud
st.header("WordCloud")

if 'review' in df_reviews.columns:
    st.write("Nube de palabras de las reseñas (ya aplicado con stopwords):")

    # Crear un texto combinado de todas las palabras limpias
    clean_text_combined = " ".join(text_stats['clean_tokens'])

    # Generar WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate(clean_text_combined)

    # Convertir la imagen a bytes para mostrarla
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

#==============================================================================================================
#==================================== Seccion 8: Distribucion de vocabulario ==================================
#==============================================================================================================

# Sección de distribución de vocabulario
st.header("Distribución de Vocabulario")

if 'review' in df_reviews.columns:
    # Contar cuántas veces aparece cada palabra
    word_counts = Counter(text_stats['clean_tokens'])

    # Contar cuántas palabras aparecen 1 vez, 2 veces, etc.
    frequency_distribution = Counter([count for word, count in word_counts.items()])

    # Ordenar por frecuencia para el gráfico
    sorted_freq = sorted(frequency_distribution.items())
    x, y = zip(*sorted_freq[:20])  # Mostrar solo las primeras 20 frecuencias

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, y)
    ax.set_xlabel("Frecuencia de aparición")
    ax.set_ylabel("Número de palabras")
    ax.set_title("Distribución de frecuencia de palabras")

    st.pyplot(fig)

    # Tabla con detalles
    st.write("Interpretación: El eje X muestra cuántas veces aparece una palabra, y el eje Y muestra cuántas palabras distintas aparecen ese número de veces.")

    dist_df = pd.DataFrame({
        'Frecuencia de aparición': x,
        'Número de palabras': y
    })
    st.dataframe(dist_df)


#==============================================================================================================
#==================================== Seccion 9: N-gramas =====================================================
#==============================================================================================================
# Sección de n-gramas
st.header("N-gramas")

if 'random_review' in st.session_state:
    clean_review = " ".join(clean_text(st.session_state.random_review, language=language))

    tab1, tab2, tab3, tab4 = st.tabs(["Bi-gramas", "Tri-gramas", "Cuatri-gramas", "Quintu-gramas"])

    with tab1:
        bigrams = get_ngrams(clean_review, n=2, language=language)
        st.write(f"Bi-gramas de la reseña aleatoria ({len(bigrams)} en total):")
        if bigrams:
            st.write(str(bigrams[:20]) + ("..." if len(bigrams) > 20 else ""))
        else:
            st.write("No se encontraron bi-gramas en esta reseña.")

    with tab2:
        trigrams = get_ngrams(clean_review, n=3, language=language)
        st.write(f"Tri-gramas de la reseña aleatoria ({len(trigrams)} en total):")
        if trigrams:
            st.write(str(trigrams[:20]) + ("..." if len(trigrams) > 20 else ""))
        else:
            st.write("No se encontraron tri-gramas en esta reseña.")

    with tab3:
        cuatrigrams = get_ngrams(clean_review, n=4, language=language)
        st.write(f"Cuatri-gramas de la reseña aleatoria ({len(cuatrigrams)} en total):")
        if cuatrigrams:
            st.write(str(cuatrigrams[:20]) + ("..." if len(cuatrigrams) > 20 else ""))
        else:
            st.write("No se encontraron cuatri-gramas en esta reseña.")

    with tab4:
        quitugrams = get_ngrams(clean_review, n=5, language=language)
        st.write(f"Quintu-gramas de la reseña aleatoria ({len(quitugrams)} en total):")
        if quitugrams:
            st.write(str(quitugrams[:20]) + ("..." if len(quitugrams) > 20 else ""))
        else:
            st.write("No se encontraron quintu-gramas en esta reseña.")



#==============================================================================================================
#==================================== Seccion 10: Expresiones regulares========================================
#==============================================================================================================
# Sección de análisis con expresiones regulares
st.header("Análisis con Expresiones Regulares")

# Agregar una sección de explicación sobre expresiones regulares
with st.expander("Guía de Expresiones Regulares", expanded=False):
    st.markdown("""
    ### Guía Rápida de Expresiones Regulares

    Las expresiones regulares (regex) son patrones utilizados para encontrar coincidencias de caracteres dentro de cadenas de texto. Aquí hay algunos patrones comunes:

    | Patrón | Descripción | Ejemplo |
    | --- | --- | --- |
    | `^` | Coincide con el inicio de la cadena | `^Monster` → "Monster Hunter" (sí), "The Monster" (no) |
    | `$` | Coincide con el final de la cadena | `Hunter$` → "Monster Hunter" (sí), "Hunter Games" (no) |
    | `..` | Coincide con cualquier carácter único | `M..nster` → "Monster", "Manster", "Mxnster" |
    | `*` | Coincide con 0 o más repeticiones del carácter anterior | `Mo*nster` → "Mnster", "Monster", "Mooonster" |
    | `+` | Coincide con 1 o más repeticiones del carácter anterior | `Mo+nster` → "Monster", "Mooonster" (pero no "Mnster") |
    | `?` | Hace que el carácter anterior sea opcional | `Monste?r` → "Monster", "Monstr" |
    | `\b` | Límite de palabra | `\\bmon\\b` → "mon" como palabra completa, no parte de otra |
    | `[]` | Conjunto de caracteres, coincide con cualquier carácter dentro de los corchetes | `[Mm]onster` → "Monster", "monster" |
    | `[^]` | Conjunto negado, coincide con cualquier carácter que NO esté dentro de los corchetes | `[^A-Z]` → cualquier carácter que no sea letra mayúscula |
    | `\\d` | Coincide con cualquier dígito | `\\d+` → cualquier número |
    | `\\w` | Coincide con cualquier carácter alfanumérico | `\\w+` → cualquier palabra |
    | `\\s` | Coincide con cualquier espacio en blanco | `\\s+` → uno o más espacios |

    ### Ejemplos prácticos:
    - `[^b]..eno..`: Encuentra palabras que inician "b" y tengan palabras de "eno"
    - `[^j]..g..[o$]`: Encuentra Palabras que contenga una "g" y termine con "o"
    """)

if 'review' in df_reviews.columns:
    col1, col2 = st.columns([3, 1])

    with col1:
        regex_pattern = st.text_input("Introduce una expresión regular", r'..eno..')

    with col2:
        search_in = st.radio("Buscar en:", ["Reseña aleatoria", "Todas las reseñas"])

    if regex_pattern:
        try:
            if search_in == "Reseña aleatoria":
                if 'random_review' in st.session_state:
                    matches = find_pattern(st.session_state.random_review, regex_pattern)
                    st.write(f"Coincidencias encontradas en la reseña aleatoria: {len(matches)}")
                    if matches:
                        st.write(matches)
                    else:
                        st.write("No se encontraron coincidencias en esta reseña.")
            else:
                all_text = " ".join(df_reviews['review'].tolist())
                matches = find_pattern(all_text, regex_pattern)
                st.write(f"Coincidencias encontradas en todas las reseñas: {len(matches)}")
                if matches:
                    if len(matches) > 100:
                        st.write(f"Mostrando las primeras 100 de {len(matches)} coincidencias:")
                        st.write(matches[:100])
                    else:
                        st.write(matches)
                else:
                    st.write("No se encontraron coincidencias.")
        except Exception as e:
            st.error(f"Error en la expresión regular: {str(e)}")


#==============================================================================================================
#==================================== Seccion 11: Limpieza (stopwords) ========================================
#==============================================================================================================
# Sección de eliminación de stopwords
st.header("Eliminación de Stopwords")

if 'random_review' in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reseña Original")
        st.write(st.session_state.random_review)

        # Tokenizar para mostrar estadísticas
        tokens_original = nltk.word_tokenize(st.session_state.random_review.lower(), language=language)
        st.write(f"Número de palabras: {len(tokens_original)}")

    with col2:
        st.subheader("Reseña sin Stopwords")
        clean_words = clean_text(st.session_state.random_review, language=language)
        clean_review = " ".join(clean_words)
        st.write(clean_review)
        st.write(f"Número de palabras: {len(clean_words)}")

    # Mostrar las stopwords que se eliminaron
    original_set = set(tokens_original)
    clean_set = set(clean_words)
    removed_words = original_set - clean_set

    stop_words = set(stopwords.words(language))
    common_stops = removed_words.intersection(stop_words)

    st.subheader("Stopwords eliminadas")
    st.write(f"Se eliminaron {len(removed_words)} palabras, de las cuales {len(common_stops)} son stopwords comunes.")

    if common_stops:
        st.write(", ".join(list(common_stops)[:30]) + ("..." if len(common_stops) > 30 else ""))

# Autores
st.header("Creado por:")
st.markdown(""" 737066 | Victor M. Telles A. | (AHTyler)""")

# Footer
st.markdown("---")
st.write("Desarrollado con Streamlit para el análisis de sentimientos en reseñas de Steam (Proyecto de lenguaje procesamiento natural)")