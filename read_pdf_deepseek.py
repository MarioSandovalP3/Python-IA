"""
Este script extrae texto de un archivo PDF y envía ese texto a la API de DeepSeek para su procesamiento.
"""

# Instalación de la librería necesaria (si no está instalada, ejecuta en la terminal):
# pip install PyPDF2 requests

# Explicación de las librerías utilizadas:
# * PyPDF2: Utilizado para leer y extraer texto de archivos PDF.
# * requests: Se utiliza para hacer solicitudes HTTP a la API de DeepSeek.

import PyPDF2
import requests

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un archivo PDF.
    
    :param pdf_path: Ruta al archivo PDF.
    :return: Texto extraído del PDF.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)  # Abre el PDF y crea un objeto lector
        text = ''
        # Iterar sobre cada página del PDF y extraer el texto
        for page in reader.pages:
            text += page.extract_text()  # Extrae el texto de cada página
    return text

# Función para enviar texto a la API de DeepSeek
def query_deepseek_api(text, api_key):
    """
    Envía el texto extraído a la API de DeepSeek para procesarlo.
    
    :param text: Texto extraído del PDF.
    :param api_key: Tu clave de API de DeepSeek.
    :return: Respuesta de la API de DeepSeek.
    """
    url = "https://api.deepseek.com/chat/completions"  # Revisa la URL correcta en la documentación de DeepSeek
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "MODELO",  # Revisa el modelo correcto en la documentación
        "messages": [
            {"role": "user", "content": f"Dame un analisis: {text}"}
        ]
    }
    # Enviar la solicitud POST a la API de DeepSeek
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    pdf_path = "documento.pdf"  # Cambia esto por la ruta de tu archivo PDF
    api_key = "TU_CLAVE_API"  # Cambia esto por tu clave de API de DeepSeek

    # Extraer texto del PDF
    print("Extrayendo texto del PDF...")
    try:
        text = extract_text_from_pdf(pdf_path)
        print("Texto extraído correctamente.")
    except Exception as e:
        print(f"Error al extraer texto del PDF: {e}")
        exit()

    # Enviar texto a la API de DeepSeek
    print("Enviando texto a la API de DeepSeek...")
    try:
        response = query_deepseek_api(text, api_key)
        print("Respuesta de la API de DeepSeek:")
        print(response)  # Muestra la respuesta completa de la API
    except Exception as e:
        print(f"Error al enviar texto a la API de DeepSeek: {e}")
