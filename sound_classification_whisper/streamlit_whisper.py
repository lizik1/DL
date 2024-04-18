import streamlit as st
import whisper
import librosa
import numpy as np
import io
from st_audiorec import st_audiorec

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
# import base64
# from streamlit.components.v1 import html
import matplotlib.pyplot as plt
from PIL import Image


st.title("Транскриптор аудио")


st.markdown("![Alt Text](https://media4.giphy.com/media/oGP14S1XBi4yk/200w.webp?cid=ecf05e474f01jb4mtj084vbrxms9z82oh00y0qwxpq88s5jd&ep=v1_gifs_related&rid=200w.webp&ct=g)")


# online record
wav_audio_data = st_audiorec()


# upload audio file with streamlit
audio_file = st.file_uploader("Загрузить файл", type=["wav", "mp3", "m4a"])


text = st.text_input("Введите текст")
tts_button = Button(label="Произнести", width=100)
tts_button.js_on_event("button_click", CustomJS(code=f"""
    var u = new SpeechSynthesisUtterance();
    u.text = "{text}";
    u.lang = 'ru';
    speechSynthesis.speak(u);
    """))
st.bokeh_chart(tts_button)



model = whisper.load_model("base")
st.success("Модель Whisper загружена")


def display_plots(audio, sr):
    st.subheader("Waveform")

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("WavePlot")

    # Save it to a temporary buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Use PIL to create an image from the new buffer
    image = Image.open(buf)
    # Display the image in Streamlit
    st.image(image, width=800)
    
    st.subheader("Spectrogram")

    # Use librosa to calculate the spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Save it to a temporary buffer again
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Create an image again
    image = Image.open(buf)
    # Display the image
    st.image(image, width=800)

    # Cleaning up figures to prevent reuse
    plt.close('all')


if st.sidebar.button("Траскрибировать аудио"):
    if audio_file is not None:
        # Read the audio file as bytes
        audio_bytes = audio_file.read()
        # Use librosa to load the audio file from the byte content
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        # If model requires single channel (mono) and audio is stereo, take only one channel
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio = librosa.to_mono(audio)
        # Convert the audio to the required format
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_numpy = np.array(audio).astype(np.float32)
        
        st.sidebar.success("Транскрибирование аудио...")
        transcription = model.transcribe(audio_numpy)
        detected_language = transcription.get('language', 'N/A')
        st.sidebar.success(f"Язык аудио был определен как: {detected_language}")
        st.sidebar.markdown(transcription["text"])
        
        display_plots(audio, sr)
       
        
    if wav_audio_data is not None:
    
        # Use librosa to load the audio file from the byte content
        audio, sr = librosa.load(io.BytesIO(wav_audio_data), sr=None)
        # If model requires single channel (mono) and audio is stereo, take only one channel
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio = librosa.to_mono(audio)
        # Convert the audio to the required format
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_numpy = np.array(audio).astype(np.float32)
        
        st.sidebar.success("Транскрибирование аудио...")
        transcription = model.transcribe(audio_numpy)
        detected_language = transcription.get('language', 'N/A')
        st.sidebar.success(f"Язык аудио был определен как: {detected_language}")
        st.sidebar.markdown(transcription["text"])
        
        display_plots(audio, sr)
        
    else:
        st.sidebar.error("Пожалуйста, загрузите аудио файл")
        
if audio_file is not None:
    st.sidebar.header("Послушать оригинальную запись")
    st.sidebar.audio(audio_file)

if wav_audio_data is not None:
    st.sidebar.header("Послушать оригинальную запись")
    st.sidebar.audio(wav_audio_data)
    
    
# my_js = """
# console.log(document.getElementById("stop"))
# """

# # Wrapt the javascript as html code
# my_html = f"<script>{my_js}</script>"


# html(my_html)
    
