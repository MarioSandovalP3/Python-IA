"""
Este script utiliza el modelo BERT de transformers para generar embeddings de un texto de entrada.
Usa el tokenizer para procesar el texto y obtener los embeddings del último estado oculto del modelo.
"""

# Instalación de la librería necesaria (si no está instalada, ejecuta en la terminal):
# pip install transformers torch

# Explicación de las librerías utilizadas:
# * transformers: Proporciona herramientas para trabajar con modelos preentrenados de NLP como BERT.
# * torch: Utilizado para manejar tensores y ejecutar los modelos de transformers.

from transformers import BertTokenizer, BertModel
import torch

# Cargar el tokenizer y el modelo preentrenado BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Texto de entrada
text = "El gato persigue al ratón"

# Tokenizar el texto de entrada (convertir a tokens que el modelo pueda procesar)
inputs = tokenizer(text, return_tensors='pt')

# Generar embeddings del texto utilizando el modelo BERT
with torch.no_grad():  # No se requiere el cálculo de gradientes para la inferencia
    outputs = model(**inputs)

# Extraer los embeddings del último estado oculto
embeddings = outputs.last_hidden_state

# Mostrar los embeddings generados
print(embeddings)
