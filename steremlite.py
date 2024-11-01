import streamlit as st
from tempfile import NamedTemporaryFile
import moviepy.editor as mp
import time
from predict_realtime import VideoDescriptionRealTime
import config
from predict_realtime import translate_to_hindi
from predict_realtime import text_to_speech

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stApp {
        background-image: url("https://your-background-image-link.com");
        background-size: cover;
    }
    h1 {
        color: #4A90E2;
        font-family: 'Arial';
    }
    .btn-custom {
        background-color: #4CAF50; 
        color: white; 
        font-size: 18px;
        border-radius: 5px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("🎥 Non-Audio Video Summarization to Audio 🎧")
st.write("Upload a video, and our AI model will summarize it and convert it into audio format!")

st.markdown(
    """
    ### How it works:
    1. Upload a video of your choice (mp4, avi, mov, mkv).
    2. The AI model will process the video, generate a text summary, and convert it into audio.
    3. Download and listen to the summarized audio file!
    """
)

video_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov", "mkv"])
model_option = st.selectbox(
    "Choose your model:",
    ("LSTM", "LSTM + GRU", "GRU")
)

st.write(f"You selected: {model_option}")


if model_option == "LSTM":
    st.write("LSTM model will be used.")

elif model_option == "LSTM + GRU":
    st.write("LSTM + GRU model will be used.")

else:
    st.write("GRU model will be used.")

model_lang = st.selectbox(
    "Choose your Language:",
    ("Hindi", "Marathi", "kannada")
)
st.write(f"You selected: {model_lang}")


if model_lang == "Hindi":
    st.write("Hindi Language  selected.")

elif model_lang == "Marathi":
    st.write("Marathi language selected ")

else:
    st.write("kannada language will be used.")
if video_file:

    st.video(video_file)

    st.write("Processing the video...")
    progress_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.02) 
        progress_bar.progress(percent_complete + 1)
    

    temp_video = NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()

    video_to_text = VideoDescriptionRealTime(config,model_option)
    sentence = video_to_text.main_test(temp_video.name)
    print(sentence)
    hindi_text = translate_to_hindi(sentence,model_lang)
    print(hindi_text)

    audio_file_name = "hindi_audio.mp3" 
    text_to_speech(hindi_text, audio_file_name,model_lang)
    
    st.audio(audio_file_name)

    with open(audio_file_name, "rb") as file:
        st.download_button(
            label="Download Audio Summary 🎧",
            data=file,
            file_name="summarized_audio.mp3",
            mime="audio/mp3",
            help="Click to download the summarized audio file."
        )

    video = mp.VideoFileClip(temp_video.name)
    st.markdown(
        f"""
        **Video Information:**
        - Duration: {video.duration:.2f} seconds
        - Resolution: {video.size[0]}x{video.size[1]} pixels
        - FPS: {video.fps} frames per second
        """
    )

st.markdown(
    """
    ---
    *Developed by Bhavesh* 
    """, 
    unsafe_allow_html=True
)
