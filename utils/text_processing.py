import nltk
import re
import os
from nltk import ngrams
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Definimos una Ruta local donde se almacenarán los recursos de nltk
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")

# Crear el directorio si no existe
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Descargar recursos de NLTK necesarios
def download_nltk_resources():
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('stopwords', download_dir=NLTK_DATA_PATH)

# Limpiar texto (eliminar palabras con stopwords y caracteres no alfabéticos)
def clean_text(text, language='spanish'):
    try:
        stop_words = set(stopwords.words(language))
    except LookupError:
        download_nltk_resources()
        stop_words = set(stopwords.words(language))

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(str(text).lower())
    clean = [w for w in words if w not in stop_words]
    return clean  # Retornamos lista de palabras limpias

# Obtener n-gramas de un texto
def get_ngrams(text, n=2, language='spanish'):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    return list(ngrams(tokens, n))

# Calcular estadísticas de texto
def get_text_stats(text, language='spanish'):
    download_nltk_resources()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    word_count = len(tokens)

    # Palabras únicas
    unique_words = set(tokens)
    unique_count = len(unique_words)

    # Eliminar stopwords para análisis de palabras comunes
    stop_words = set(stopwords.words(language))
    clean_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Palabras comunes
    word_freq = Counter(clean_tokens)
    most_common_words = word_freq.most_common(10)

    return {
        'word_count': word_count,
        'unique_count': unique_count,
        'most_common': most_common_words,
        'clean_tokens': clean_tokens
    }

# Buscar patrones con regex
def find_pattern(text, pattern):
    return re.findall(pattern, text)