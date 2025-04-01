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
