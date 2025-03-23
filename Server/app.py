import streamlit as st
import requests

URL = "http://127.0.0.1:8000/predict/"

label_mapping = {
        0 : "Anger", 1 : "Fear", 2 : "Sad", 3 : "Distress",
        4 : "Surprised", 5 : "Happy", 6 : "Sad", 7 : "Neutral"
    }

st.title("Audio Sentiment classification demo")
st.markdown("##### Base model used : Hubert-base-ls960")
st.markdown("##### Classification architecture type : Mean pooling")
st.markdown("You can either upload a .wav file or record your own voice to classify the sentiment.")

file_upload = st.file_uploader("Choose yout .wav file", type=["wav"])
audio_value = st.audio_input("Record your voice message")

if file_upload and audio_value:
    st.warning("You can either upload or record can't do both !!")

elif file_upload or audio_value:

    if audio_value:
        files = {"file": audio_value}
    else:
        files = {"file" : file_upload}

    response = requests.post(URL, files=files)
    print(response)
    st.write(f"Predicted class : {response.json()['class']}")
    st.write(f"Predicted sentiment : {label_mapping[response.json()['class']]}")
    st.write(f"Confidence : {response.json()['confidence']}")
