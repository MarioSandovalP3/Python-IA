"""
Este script utiliza la API de OpenAI para generar respuestas a partir de una 
consulta. Envía un mensaje a un modelo de lenguaje y devuelve la respuesta generada.
"""

# Instalación de la librería necesaria (si no está instalada, ejecuta en la terminal):
# pip install openai

# Explicación de la librería utilizada:
# * openai: Permite interactuar con la API de OpenAI para generar respuestas basadas en texto.

import openai

# Configuración del cliente de OpenAI con la clave API y la URL base de la API
client = openai.OpenAI(api_key="TU_CLAVE_API", base_url="BASE_URL_DE_LA_API")

def generar_respuesta(prompt):
    """
    Genera una respuesta basada en un mensaje de entrada utilizando la API de OpenAI.
    
    :param prompt: Texto de entrada con la consulta del usuario.
    :return: Respuesta generada por el modelo o un mensaje de error.
    """
    try:
        response = client.chat.completions.create(
            model="MODELO",  # Especifica el modelo a utilizar, por ejemplo, "gpt-4"
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Ejemplo de uso
if __name__ == "__main__":
    pregunta = "Explica el concepto de inteligencia artificial de manera sencilla."
    respuesta = generar_respuesta(pregunta)
    print("Respuesta:", respuesta)


