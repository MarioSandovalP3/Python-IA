"""
Este script lee el contenido de un archivo PDF, lo divide en fragmentos, 
genera embeddings para cada fragmento usando un modelo de Sentence Transformers 
y luego encuentra el fragmento más relevante en función de una consulta 
utilizando la similitud coseno.
"""

# Instalación de librerías necesarias (si no están instaladas, ejecuta este comando en la terminal):
# pip install sentence-transformers scikit-learn numpy pymupdf

# Explicación de las librerías utilizadas:
# * sentence-transformers: Para cargar y generar embeddings con modelos preentrenados.
# * scikit-learn: Proporciona la función cosine_similarity para calcular la similitud coseno.
# * numpy: Se usa para manipular arrays y encontrar el índice del documento más relevante.
# * pymupdf (también llamado fitz): Se utiliza para leer archivos PDF y extraer su contenido.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF

# Cargar un modelo preentrenado para generar embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Función para leer el contenido de un archivo PDF
def leer_pdf(ruta_pdf):
    # Abrir el archivo PDF
    doc = fitz.open(ruta_pdf)
    texto = ""
    # Iterar sobre todas las páginas del PDF
    for pagina in doc:
        texto += pagina.get_text()
    return texto

# Cargar y leer el PDF
ruta_pdf = "documento.pdf"  # Reemplaza con el nombre del archivo PDF
contenido_pdf = leer_pdf(ruta_pdf)

# Dividir el contenido en fragmentos (puedes usar otro tipo de división si prefieres)
documentos = contenido_pdf.split("\n")  # O cualquier otro método de división que prefieras

# Generar embeddings para los documentos (fragmentos del PDF)
embeddings_documentos = model.encode(documentos)

# Consulta de ejemplo
consulta = "¿De qué trata el PDF?"  # Puedes personalizar esta pregunta
embedding_consulta = model.encode([consulta])

# Calcular similitud coseno
similitudes = cosine_similarity(embedding_consulta, embeddings_documentos)
indice_relevante = np.argmax(similitudes)

documento_relevante = documentos[indice_relevante]
print(f"Documento más relevante: {documento_relevante}")
