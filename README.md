# <img src="https://www.linkve.net/storage/images/python-ia.png" alt="Logo"  height="100"> Python + IA

Este repositorio contiene varios scripts de Python que implementan técnicas de inteligencia artificial (IA) para procesar y analizar textos, así como interactuar con APIs de IA. Cada script tiene un enfoque diferente para demostrar diversas aplicaciones de IA, desde el análisis de texto hasta la integración con modelos preentrenados.

## Scripts

### 1. **[ApiChat](https://github.com/MarioSandovalP3/Python-IA/blob/main/api_chat.py)**
Este script utiliza la API de OpenAI para generar respuestas a partir de una 
consulta. Envía un mensaje a un modelo de lenguaje y devuelve la respuesta generada. 

**Dependencias:**
- [`openai`](https://pypi.org/project/openai/0.26.5/)

**Explicación de la librería utilizada:**
* openai: Permite interactuar con la API de OpenAI para generar respuestas basadas en texto.

**Instalación:**

```python
pip install openai
```

### 2. **[BertEmbeddings](https://github.com/MarioSandovalP3/Python-IA/blob/main/bert_embeddings.py)**
Este script utiliza el modelo BERT de transformers para generar embeddings de un texto de entrada.
Usa el tokenizer para procesar el texto y obtener los embeddings del último estado oculto del modelo.

**Dependencias:**
- [`transformers`](https://pypi.org/project/transformers/)
- [`torch`](https://pypi.org/project/torch/)

**Explicación de las librerís utilizadas:**
* transformers: Proporciona herramientas para trabajar con modelos preentrenados de NLP como BERT.
* torch: Utilizado para manejar tensores y ejecutar los modelos de transformers.

**Instalación:**

```python
pip install transformers torch
```