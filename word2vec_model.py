"""
Este script entrena un modelo de Word2Vec usando el corpus de ejemplo 
y luego obtiene el vector de una palabra específica ('gato') del modelo entrenado.
"""

# Instalación de la librería necesaria (si no está instalada, ejecuta en la terminal):
# pip install gensim

# Explicación de las librerías utilizadas:
# * gensim: Utilizado para entrenar modelos de Word2Vec y trabajar con representaciones vectoriales de palabras.

from gensim.models import Word2Vec

# Corpus de ejemplo: cada oración es una lista de palabras
sentences = [["el", "gato", "persigue", "al", "ratón"],
             ["el", "perro", "juega", "con", "el", "gato"]]

# Entrenar el modelo Word2Vec
# vector_size: tamaño de los vectores de las palabras
# window: el contexto de palabras alrededor de la palabra objetivo
# min_count: número mínimo de ocurrencias de la palabra en el corpus para ser considerada
# workers: número de hilos para entrenar el modelo
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Obtener el vector de una palabra específica ('gato')
vector_gato = model.wv['gato']

# Imprimir el vector de la palabra 'gato'
print(vector_gato)
