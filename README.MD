# Análisis de Sentimientos en Reseñas de Steam
Este proyecto utiliza Streamlit y NLTK para analizar reseñas de videojuegos obtenidas de la API de Steam, centrándose en el procesamiento de lenguaje natural y análisis de sentimientos.
Características

- Obtención de reseñas de juegos de Steam a través de su API
- Análisis de sentimientos basado en las calificaciones positivas/negativas
- Visualización de estadísticas de texto (palabras más comunes, únicas, etc.)
- Generación de WordClouds para visualizar términos frecuentes
- Extracción y visualización de N-gramas (bi-gramas, tri-gramas, etc.)
- Análisis mediante expresiones regulares
- Eliminación de stopwords para mejorar el análisis

# Requisitos
Consulta el archivo requirements.txt para ver todas las dependencias necesarias.

# Instalación
1. Clona este repositorio:
```bash
git clone https://github.com/victortelles/Stream_Analisis_Sentimiento_ReviewsGames
cd Stream_Analisis_Sentimiento_ReviewsGames
```

2. Creacion de Ambiente de entorno de python
```bash
python -m venv venv
```

3. Activar el entorno
```bash
cd venv/Scripts/Activate.ps1
```

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

5. Ejecutar la aplicacion:
```bash
streamlit run app.py
```

# Uso

1. Ingresa el ID del juego de Steam que deseas analizar (por ejemplo, Monster Hunter Wilds = 2246340)
2. Selecciona el idioma de las reseñas
3. Ajusta el número de reseñas a obtener
4. Haz clic en "Obtener Reseñas" para iniciar el análisis