import os
from focal_loss import MyBinaryFocalCrossentropy   
import re
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences



@st.cache_resource
def load_model_and_assets():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    model = tf.keras.models.load_model(
        os.path.join(root_path, "lstm_genre_model.h5"),
        custom_objects={"MyBinaryFocalCrossentropy": MyBinaryFocalCrossentropy}
    )

    with open(os.path.join(root_path, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(root_path, "mlb.pkl"), "rb") as f:
        mlb = pickle.load(f)

    thresholds_path = os.path.join(root_path, "thresholds.npy")
    thresholds = np.load(thresholds_path)

    return model, tokenizer, mlb, thresholds



def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_genres(text, model, tokenizer, mlb, thresholds):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    probs = model.predict(padded)[0]  # shape: (n_genres,)

    preds = (probs >= thresholds).astype(int)
    margins = probs - thresholds
    genres = mlb.inverse_transform(np.array([preds]))[0]

    return genres, probs, thresholds, margins

st.set_page_config(page_title="Song Genre Classifier", layout="centered")
st.title("Song Genre Classifier")

user_input = st.text_area("Enter song lyrics:", height=250)

if st.button("Predict Genre"):
    model, tokenizer, mlb, thresholds = load_model_and_assets()
    if user_input.strip():
        genres, probs, thresholds, margins = predict_genres(user_input, model, tokenizer, mlb, thresholds)
        st.subheader("Predicted Genres (by confidence):")
        
        genre_info = []
        for i, genre in enumerate(mlb.classes_):
            prob = float(probs[i])
            threshold = float(thresholds[i])
            margin = prob - threshold
            genre_info.append((genre, prob, threshold, margin))
        
        genre_info.sort(key=lambda x: x[3], reverse=True)

        confident_genres = [g for g in genre_info if g[1] >= g[2]]

        if confident_genres:
            for genre, prob, threshold, margin in confident_genres:
                st.markdown(f"- *{genre}* (Prob: {prob:.2f}, Threshold: {threshold:.2f}, Margin: {margin:+.2f})")
        else:
            st.warning("No genre confidently predicted. Try with more or clearer lyrics.")

        st.subheader("All Genre Probabilities and Margins:")
        for genre, prob, threshold, margin in genre_info:
            st.progress(prob, text=f"{genre}: {prob:.2f} (Threshold: {threshold:.2f}, Margin: {margin:+.2f})")


    else:
        st.error("Please enter lyrics before clicking predict.")