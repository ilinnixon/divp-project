

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import tempfile
import os

st.title("🎥 Shoplifting Detection Dashboard (I3D + LSTM v3)")
st.write("Upload a video clip to detect shoplifting behavior using the trained I3D-LSTM model.")

# -----------------------------------------------------------
# Load Models
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    # I3D feature extractor (Kinetics-400 pretrained)
    i3d_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    i3d_layer = hub.KerasLayer(i3d_url, trainable=False, name="i3d_base")

    # Your trained LSTM classifier
    lstm_model = tf.keras.models.load_model("shoplifting_i3d_lstm_v3.keras")
    return i3d_layer, lstm_model

i3d_layer, lstm_model = load_models()

# -----------------------------------------------------------
# Upload video
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    # -----------------------------------------------------------
    # Frame Extraction
    # -----------------------------------------------------------
    st.info("🎞️ Extracting and preprocessing frames...")
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    st.write(f"✅ Extracted {total_frames} frames.")

    if total_frames < 16:
        st.error("⚠️ Please upload a longer video (at least 16 frames).")
    else:
        frames = np.array(frames) / 255.0

        # Pad or trim to 64 frames
        max_frames = 64
        if total_frames > max_frames:
            frames = frames[:max_frames]
        elif total_frames < max_frames:
            pad_len = max_frames - total_frames
            pad_frames = np.zeros((pad_len, 224, 224, 3))
            frames = np.concatenate([frames, pad_frames], axis=0)

        frames = np.expand_dims(frames, axis=0)  # (1, 64, 224, 224, 3)

        # -----------------------------------------------------------
        # Extract I3D features (400-dim)
        # -----------------------------------------------------------
        st.info("🔍 Extracting I3D features...")
        features = i3d_layer(frames).numpy()  # (1, 400)
        features = np.expand_dims(features, axis=1)      # (1, 1, 400)

        # -----------------------------------------------------------
        # Run classification
        # -----------------------------------------------------------
        st.info("🤖 Classifying video with LSTM model...")
        preds = lstm_model.predict(features, verbose=0)
        score = float(preds[0][0])

        label = "✅ Normal Activity" if score > 0.5 else "🚨 Shoplifting Detected"
        confidence = score if score > 0.5 else 1 - score

        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

    try:
        os.remove(tfile.name)
    except PermissionError:
        pass
