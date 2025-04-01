import nltk
import re
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter

# Descargar recursos de NLTK necesarios
def download_nltk_resources():
    resources = ['stopwords', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Obtener n-gramas de un texto
def get_ngrams(text, n=2, language='spanish'):
    tokens = nltk.word_tokenize(text.lower(), language=language)
    return list(ngrams(tokens, n))

# Calcular estadísticas de texto
def get_text_stats(text, language='spanish'):
    download_nltk_resources()
    tokens = nltk.word_tokenize(text.lower(), language=language)
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
