# Python + IA

Este repositorio contiene varios scripts de Python que implementan técnicas de inteligencia artificial (IA) para procesar y analizar textos, así como interactuar con APIs de IA. Cada script tiene un enfoque diferente para demostrar diversas aplicaciones de IA, desde el análisis de texto hasta la integración con modelos preentrenados.

## Scripts

### 1. **[ApiChat](https://github.com/MarioSandovalP3/Python-IA/blob/main/api_chat.py)**
Este script utiliza la API de OpenAI para generar respuestas a partir de una 
consulta. Envía un mensaje a un modelo de lenguaje y devuelve la respuesta generada. 

**Dependencias:**
- [`openai`](https://pypi.org/project/openai/0.26.5/)

**Explicación de la librería utilizada:**
* **openai:** Permite interactuar con la API de OpenAI para generar respuestas basadas en texto.

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
* **transformers:** Proporciona herramientas para trabajar con modelos preentrenados de NLP como BERT.
* **torch:** Utilizado para manejar tensores y ejecutar los modelos de transformers.

**Instalación:**

```python
pip install transformers torch
```


### 3. **[PDFSimilaritySearch](https://github.com/MarioSandovalP3/Python-IA/blob/main/pdf_similarity_search.py)**
Este script lee el contenido de un archivo PDF, lo divide en fragmentos, 
genera embeddings para cada fragmento usando un modelo de Sentence Transformers 
y luego encuentra el fragmento más relevante en función de una consulta 
utilizando la similitud coseno.

**Dependencias:**
- [`sentence-transformers`](https://pypi.org/project/sentence-transformers/)
- [`scikit-learn`](https://pypi.org/project/scikit-learn/)
- [`numpy`](https://pypi.org/project/numpy/)
- [`pymupdf`](https://pypi.org/project/PyMuPDF/)

**Explicación de las librerís utilizadas:**
* **sentence-transformers:** Para cargar y generar embeddings con modelos preentrenados.
* **scikit-learn:** Proporciona la función cosine_similarity para calcular la similitud coseno.
* **numpy:** Se usa para manipular arrays y encontrar el índice del documento más relevante.
* **pymupdf (también llamado fitz):** Se utiliza para leer archivos PDF y extraer su contenido.

**Instalación:**

```python
pip install sentence-transformers scikit-learn numpy pymupdf
```


### 4. **[RagSimilarity](https://github.com/MarioSandovalP3/Python-IA/blob/main/rag_similarity.py)**
Este script utiliza embeddings de texto para encontrar el documento más relevante 
según una consulta. Se basa en la similitud coseno entre los embeddings de la 
consulta y los documentos predefinidos.

**Dependencias:**
- [`sentence-transformers`](https://pypi.org/project/sentence-transformers/)
- [`scikit-learn`](https://pypi.org/project/scikit-learn/)
- [`numpy`](https://pypi.org/project/numpy/)

**Explicación de las librerís utilizadas:**
* **sentence-transformers:** Para cargar y generar embeddings con modelos preentrenados.
* **scikit-learn:** Proporciona la función cosine_similarity para calcular la similitud coseno.
* **numpy:** Se usa para manipular arrays y encontrar el índice del documento más relevante.

**Instalación:**

```python
pip install sentence-transformers scikit-learn numpy
```


### 5. **[ReadPdfDeepseek](https://github.com/MarioSandovalP3/Python-IA/blob/main/read_pdf_deepseek.py)**
Este script extrae texto de un archivo PDF y envía ese texto a la API de DeepSeek para su procesamiento.

**Dependencias:**
- [`PyPDF2`](https://pypi.org/project/PyPDF2/)
- [`requests`](https://pypi.org/project/requests/)

**Explicación de las librerís utilizadas:**
* **PyPDF2:** Utilizado para leer y extraer texto de archivos PDF.
* **requests:** Se utiliza para hacer solicitudes HTTP a la API de DeepSeek.

**Instalación:**

```python
pip install PyPDF2 requests
```


### 6. **[Word2vecModel](https://github.com/MarioSandovalP3/Python-IA/blob/main/word2vec_model.py)**
Este script entrena un modelo de Word2Vec usando el corpus de ejemplo 
y luego obtiene el vector de una palabra específica ('gato') del modelo entrenado.

**Dependencias:**
- [`gensim`](https://pypi.org/project/gensim/)

**Explicación de las librerís utilizadas:**
* **gensim:** Utilizado para entrenar modelos de Word2Vec y trabajar con representaciones vectoriales de palabras.

**Instalación:**

```python
pip install gensim
```