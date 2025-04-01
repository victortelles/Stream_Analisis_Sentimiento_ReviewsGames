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
