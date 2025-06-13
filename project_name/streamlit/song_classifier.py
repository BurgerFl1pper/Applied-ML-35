import os
import re
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_and_assets():
    """
    Loads the pre-trained LSTM model and associated assets (tokenizer, MultiLabelBinarizer, thresholds).
    Uses Streamlit's caching to avoid reloading on every interaction.

    Returns:
        model (tf.keras.Model): The loaded Keras model.
        tokenizer (Tokenizer): Tokenizer used during training.
        mlb (MultiLabelBinarizer): Label binarizer for genre labels.
        thresholds (np.ndarray): Optimal thresholds for each genre.
    """
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Load model
    model_path = os.path.join(root_path, "lstm_genre_model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load tokenizer
    tokenizer_path = os.path.join(root_path, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load MultiLabelBinarizer
    mlb_path = os.path.join(root_path, "mlb.pkl")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    # load thresholds
    thresholds_path = os.path.join(root_path, "thresholds.npy")
    thresholds = np.load(thresholds_path)

    return model, tokenizer, mlb, thresholds

def clean_text(text):
    """
    Cleans input text by lowercasing, removing unwanted characters, and normalizing whitespace.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\[.*?\]", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_genres(text, model, tokenizer, mlb, thresholds):
    """
    Predicts genres for a given text input using the trained model.

    Args:
        text (str): Raw input text.
        model (tf.keras.Model): Loaded LSTM model.
        tokenizer (Tokenizer): Trained tokenizer.
        mlb (MultiLabelBinarizer): Trained label binarizer.
        thresholds (np.ndarray): Class-specific thresholds for multi-label decision.

    Returns:
        tuple: (predicted genres, probabilities, thresholds, margins)
    """
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
    probs = model.predict(padded, verbose=0)[0]  
    preds = (probs >= thresholds).astype(int)
    margins = probs - thresholds
    genres = mlb.inverse_transform(np.array([preds]))[0]

    return genres, probs, thresholds, margins

# Streamlit UI
st.set_page_config(page_title="Song Genre Classifier", layout="centered")
st.title("Song Genre Classifier")

# load the model and assets
model, tokenizer, mlb, thresholds = load_model_and_assets()

# user can put there lyrics in this box
user_input = st.text_area("Enter song lyrics:", height=250)

if st.button("Predict Genre"):
    if user_input.strip():
        genres, probs, thresholds, margins = predict_genres(user_input, model, tokenizer, mlb, thresholds)

        st.subheader("Predicted Genres:")
        if genres:
            st.write(", ".join(genres))
        else:
            st.warning("No genre confidently predicted. Try with more lyrics.")

        st.subheader("Genre Probabilities and Margins:")
        genre_info = []
        for i, genre in enumerate(mlb.classes_):
            prob = float(probs[i])
            threshold = float(thresholds[i])
            margin = prob - threshold
            genre_info.append((genre, prob, threshold, margin))

        genre_info.sort(key=lambda x: x[3], reverse=True)

        for genre, prob, threshold, margin in genre_info:
            st.progress(prob, text=f"{genre}: {prob:.2f} (Threshold: {threshold:.2f}, Margin: {margin:+.2f})")

    else:
        st.error("Please enter lyrics before clicking predict.")