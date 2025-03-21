"""
Este script utiliza embeddings de texto para encontrar el documento más relevante 
según una consulta. Se basa en la similitud coseno entre los embeddings de la 
consulta y los documentos predefinidos.
"""

# Instalación de librerías necesarias (si no están instaladas, ejecuta en la terminal):
# pip install sentence-transformers scikit-learn numpy

# Explicación de las librerías utilizadas:
# * sentence-transformers: Para cargar y generar embeddings con modelos preentrenados.
# * scikit-learn: Proporciona la función cosine_similarity para calcular la similitud coseno.
# * numpy: Se usa para manipular arrays y encontrar el índice del documento más relevante.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cargar un modelo preentrenado para generar embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Crear un índice de documentos
documentos = [
    "Python es un lenguaje de programación versátil.",
    "RAG combina recuperación y generación de texto.",
    "Los embeddings son representaciones vectoriales de texto."
]

# Generar embeddings para los documentos
embeddings_documentos = model.encode(documentos)

# Consulta de ejemplo
consulta = "¿Qué es RAG?"
embedding_consulta = model.encode([consulta])

# Calcular similitud coseno
similitudes = cosine_similarity(embedding_consulta, embeddings_documentos)
indice_relevante = np.argmax(similitudes)

# Obtener el documento más relevante según la similitud
documento_relevante = documentos[indice_relevante]
print(f"Documento más relevante: {documento_relevante}")
