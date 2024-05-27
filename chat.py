import streamlit as st
import openai
import json
import nltk
import os
import tempfile
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import PyPDF2
import time
from google.cloud import texttospeech
from streamlit_webrtc import webrtc_streamer, WebRtcMode

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

# Cargar credenciales desde Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Verificar si el JSON es válido
google_creds_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
try:
    google_creds_dict = json.loads(google_creds_json)
except json.JSONDecodeError:
    st.error("Las credenciales de Google no son un JSON válido.")
    google_creds_dict = None

if google_creds_dict:
    # Guardar las credenciales de Google en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        json.dump(google_creds_dict, temp_file)
        temp_file_path = temp_file.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

# Instancia el cliente de Text-to-Speech si las credenciales son válidas
if google_creds_dict:
    client = texttospeech.TextToSpeechClient()

# Función para obtener respuesta de OpenAI usando el modelo GPT y convertir a audio
def obtener_respuesta(pregunta, texto_preprocesado, modelo, temperatura=0.5):
    try:
        response = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Actua como Ana la asesora de ventas del restaurante Sazon Burguer y resuelve las inquietudes de los clientes, tienes un tono muy amable y cordial"},
                {"role": "user", "content": f"{pregunta}\n\nContexto: {texto_preprocesado}"}
            ],
            temperature=temperatura
        )
        respuesta = response.choices[0].message['content'].strip()

        # Configura la solicitud de síntesis de voz
        input_text = texttospeech.SynthesisInput(text=respuesta)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Realiza la solicitud de síntesis de voz
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )

        # Reproduce el audio en Streamlit
        st.audio(response.audio_content, format="audio/mp3")
        return respuesta

    except openai.OpenAIError as e:
        st.error(f"Error al comunicarse con OpenAI: {e}")
        return "Lo siento, no puedo procesar tu solicitud en este momento."

def main():
    # --- Diseño general ---
    st.set_page_config(page_title="SAZON BURGUER", page_icon="🤖")

    # --- Barra lateral ---
    with st.sidebar:
        st.image("hamburguesa.jpg")
        st.title("🤖 RESTAURANTE SAZON BURGUER")
        st.markdown("---")
        # --- Botones de historial ---
        if st.button("Buscar Historial"):
            st.session_state.mostrar_historial = True
        if st.button("Borrar Historial"):
            st.session_state.mensajes = []
            st.session_state.mostrar_historial = False
            st.success("Historial borrado correctamente")
    
    # --- Chatbot ---
    if 'mensajes' not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    # Función para manejar la entrada de audio
    def on_audio(audio_bytes):
        with st.spinner("Transcribiendo..."):
            transcript = openai.Audio.transcribe("whisper-1", audio_bytes)
            pregunta_usuario = transcript["text"]
            st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario, "timestamp": time.time()})
            with st.chat_message("user"):
                st.markdown(pregunta_usuario)

    st.subheader("🎤 Captura de voz")
    st.info("Haz clic en el micrófono y comienza a hablar. Tu pregunta se transcribirá automáticamente.")
    with st.container():
        if st.button("Grabar 🎙️"):
            st.session_state.run_webrtc = True
        if st.session_state.get("run_webrtc", False):
            webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
                on_audio=on_audio,
            )
            st.session_state["run_webrtc"] = False

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    # Selección de modelo de lenguaje
    st.subheader("🧠 Configuración del Modelo")
    modelo = st.selectbox(
        "Selecciona el modelo:",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="Elige el modelo de lenguaje de OpenAI que prefieras."
    )

    # --- Opciones adicionales ---
    st.markdown("---")
    temperatura = st.slider("🌡️ Temperatura", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # --- Video de fondo ---
    with st.container():
        st.markdown(
            f"""
            <style>
            #video-container {{
                position: relative;
                width: 100%;
                padding-bottom: 56.25%;
                background-color: lightblue;
                overflow: hidden;
            }}
            #background-video {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
            </style>
            <div id="video-container">
                <video id="background-video" autoplay loop muted playsinline>
                    <source src="https://cdn.leonardo.ai/users/645c3d5c-ca1b-4ce8-aefa-a091494e0d09/generations/89dda365-bf17-4867-87d4-bd918d4a2818/89dda365-bf17-4867-87d4-bd918d4a2818.mp4" type="video/mp4">
                </video>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Área principal de la aplicación ---
    st.header("💬 Hablar con Ana asesor")

    # Carga de archivo PDF
    archivo_pdf = st.file_uploader("📂 Cargar PDF", type='pdf')

    pregunta_usuario = st.chat_input("Pregunta:")
    if pregunta_usuario:
        st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        with st.spinner("Ana está pensando..."):  # Mostrar spinner de carga
            if archivo_pdf:
                texto_pdf = extraer_texto_pdf(archivo_pdf)
                texto_preprocesado = preprocesar_texto(texto_pdf)
            else:
                texto_preprocesado = ""  # Sin contexto de
