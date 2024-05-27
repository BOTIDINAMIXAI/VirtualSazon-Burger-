REQUERIMENTS
se deben instalalar:
streamlit
openai
python-dotenv
nltk
PyPDF2
google-cloud-texttospeech
streamlit-webrtc
streamlit-3d-model # Si estás utilizando esta biblioteca para el modelo 3D

Instalar antes de ejecutar 

pip install --upgrade pip
pip install -r requirements.txt

Pasos para ejecutar el programa:
# Chatbot con OpenAI

Este proyecto consiste en un chatbot desarrollado utilizando la API de OpenAI. El chatbot es capaz de responder preguntas sobre un texto proporcionado por el usuario, utilizando modelos de lenguaje avanzados de OpenAI.

## Descripción

El chatbot puede cargar un documento PDF, extraer su contenido de texto, preprocesarlo y luego responder preguntas sobre ese texto utilizando la API de OpenAI. Los usuarios pueden seleccionar diferentes modelos de lenguaje de OpenAI para generar respuestas.

## Instalación

Para ejecutar este proyecto localmente, sigue estos pasos:

1. Clona este repositorio en tu máquina local utilizando Git:

    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    ```

2. Navega al directorio del proyecto:

    ```bash
    cd tu_repositorio
    ```

3. Instala las dependencias necesarias utilizando pip:

    ```bash
    pip install -r requirements.txt
    ```

4. Configura tu clave de API de OpenAI. Puedes hacerlo estableciendo la variable de entorno `OPENAI_API_KEY` o directamente en el archivo `chat.py`.

## Uso

Una vez que hayas configurado el proyecto y las dependencias, puedes ejecutar el chatbot utilizando el siguiente comando:

```bash
streamlit run chat.py
```
## Dependencias

Este proyecto utiliza las siguientes dependencias de Python:

- Streamlit: Para la interfaz de usuario interactiva.
- OpenAI: Para acceder a la API de OpenAI y obtener respuestas del chatbot.
- NLTK: Para el preprocesamiento de texto, incluyendo tokenización y eliminación de palabras vacías.
- PyPDF2: Para extraer texto de documentos PDF.

Puedes encontrar las versiones específicas de las dependencias en el archivo `requirements.txt`.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir a este proyecto, por favor, abre un issue para discutir los cambios propuestos antes de enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo `LICENSE`.
