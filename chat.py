import streamlit as st
import openai
from dotenv import load_dotenv
import nltk
import os
import tempfile
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import PyPDF2
from google.cloud import texttospeech

# Configuración de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Función para cargar el texto del PDF
def extraer_texto_pdf(archivo):
    texto = ""
    if archivo:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(archivo.read())
            temp_file_path = temp_file.name
        with open(temp_file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in range(len(reader.pages)):
                texto += reader.pages[page].extract_text()
        os.unlink(temp_file_path)
    return texto

# Función para preprocesar texto
def preprocesar_texto(texto):
    tokens = word_tokenize(texto, language='spanish')
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stopwords_es = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stopwords_es]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Cargar la clave API desde el archivo .env
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "botidinamix-g.json"  # Reemplaza 'botidinamix-g.json' con el nombre de tu archivo de credenciales

# Cargar credenciales desde Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Instancia el cliente de Text-to-Speech
client = texttospeech.TextToSpeechClient()

# Función para obtener respuesta de OpenAI usando el modelo GPT
def obtener_respuesta(pregunta, agente, texto_preprocesado, modelo, temperatura=0.5, top_p=1.0):
    try:
        response = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": f"Eres Ana y trabajas en el restaurante Sazon Burguer, actúa como {agente} y resuelve las inquietudes de los clientes, tienes un tono muy amable y cordial, puedes utilizar emojis"},
                {"role": "user", "content": f"{pregunta}\n\nContexto: {texto_preprocesado}"}
            ],
            temperature=temperatura,
            top_p=top_p
        )
        respuesta = response.choices[0].message['content'].strip()
        return respuesta

    except openai.OpenAIError as e:
        st.error(f"Error al comunicarse con OpenAI: {e}")
        return "Lo siento, no puedo procesar tu solicitud en este momento."

def reproducir_audio(texto):
    input_text = texttospeech.SynthesisInput(text=texto)
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-ES", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return response.audio_content

def main():
    # --- Diseño general ---
    st.set_page_config(page_title="SAZON BURGUER RESTAURANTE", page_icon="🤖")



    # --- Barra lateral ---
    with st.sidebar:
        st.title("🤖 RESTAURANTE SAZON BURGUER")
        st.markdown('<p style="color:green;">Brindamos la mejor atención</p>', unsafe_allow_html=True)
        st.markdown("---")

        # Selección de agente
        agente = st.radio(
            "Selecciona el agente:",
            ["Asistente de atención al cliente", "Agente Administrativo"],
            index=0,
            help="Elige el agente con el que quieres interactuar."
        )

        # Selección de modelo de lenguaje
        st.subheader("🧠 Configuración del Modelo")
        modelo = st.selectbox(
            "Selecciona el modelo:",
            ["gpt-3.5-turbo", "gpt-4","gpt-4-32k","gpt-4o"],
            index=1,
            help="Elige el modelo de lenguaje de OpenAI que prefieras."
        )

        # --- Opciones adicionales ---
        st.markdown("---")
        temperatura = st.slider("🌡️ Temperatura", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        top_p = st.slider("🎨 Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

        # --- Historial de conversaciones ---
        st.markdown("---")
        st.subheader("🗂️ Historial de Conversaciones")
        if 'mensajes' not in st.session_state:
            st.session_state['mensajes'] = []

        historial_conversaciones = st.session_state['mensajes']
        historial_opciones = [f"Conversación {i+1}" for i in range(len(historial_conversaciones))]
        seleccion_historial = st.selectbox("Selecciona una conversación anterior:", ["Seleccionar"] + historial_opciones)

    # --- Video de fondo ---
    video_placeholder = st.empty()
    video_html = """
        <div id="video-container">
            <video id="background-video" autoplay loop muted playsinline>
                <source src="https://cdn.leonardo.ai/users/645c3d5c-ca1b-4ce8-aefa-a091494e0d09/generations/dd8e0b28-efa4-4937-aaab-a1a8ffa47568/dd8e0b28-efa4-4937-aaab-a1a8ffa47568.mp4" type="video/mp4">
            </video>
        </div>
    """
    video_placeholder.markdown(video_html, unsafe_allow_html=True)

    # --- Entrada de usuario y manejo de la conversación ---
    st.markdown("## 💬 HABLAR CON EL AGENTE")
    with st.form(key='chat_form'):
        input_usuario = st.text_area("Escribe tu mensaje:", key="input_usuario", height=80)
        submit_button = st.form_submit_button(label='Enviar')

    if submit_button and input_usuario:
        
       st.markdown("---")

if __name__ == "__main__":
    main()
