import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# =========================================
# TITLE
# =========================================

st.title("🎤 Speech Emotion Recognition")

st.write(
    "Upload a voice recording and detect emotion using AI"
)

# =========================================
# LOAD MODEL
# =========================================

with open('./real_model.pkl', 'rb') as file:
    model = pickle.load(file)

# =========================================
# EMOTIONS
# =========================================

emotions = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# =========================================
# MFCC EXTRACTION
# =========================================

def extract_mfcc(filename):

    y, sr = librosa.load(
        filename,
        duration=3,
        offset=0.5
    )

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    return mfcc

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload WAV Audio File",
    type=['wav']
)

# =========================================
# PROCESS AUDIO
# =========================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Detect Emotion"):

        with st.spinner("Analyzing Voice..."):

            # temp save
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav'
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())

                temp_path = tmp_file.name

            # extract mfcc
            mfcc = extract_mfcc(temp_path)

            mfcc = np.array(mfcc)

            # reshape
            mfcc = mfcc.reshape(1,-1)

            # prediction
            prediction = model.predict(mfcc)

            # highest probability index
            predicted_index = np.argmax(prediction)

            # emotion name
            predicted_emotion = emotions[predicted_index]

            # show result
            st.success(
                f"🎯 Predicted Emotion: {predicted_emotion.upper()}"
            )

