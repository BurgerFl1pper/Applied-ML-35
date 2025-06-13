from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'API')

VECTORIZER = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
MLB = joblib.load(os.path.join(MODEL_DIR, 'mlb.joblib'))
CLASSIFIER = joblib.load(os.path.join(MODEL_DIR, 'classifier.joblib'))


class LyricsInput(BaseModel):
    """
    Request model representing input lyrics text.

    Attributes:
        lyrics (str): The song lyrics text that will be classified.
    """
    lyrics: str

class GenrePrediction(BaseModel):
    """
    Response model representing predicted music genres.

    Attributes:
        predicted_genres (list[str]): List of predicted genre labels.
    """
    predicted_genres: list[str]


@app.post("/predict")
def predict_genres(input: LyricsInput):
    """
    Predicts music genres for the given lyrics text.

    Args:
        input (LyricsInput): Request body containing lyrics string.

    Raises:
        HTTPException: If input lyrics is empty/blank.

    Returns:
        dict: Dictionary with key 'predicted_genres' mapping to a list of predicted genre labels.
    """

    if not input.lyrics or input.lyrics.strip() == "":
        raise HTTPException(status_code=400, detail="Lyrics input cannot be empty.")

    lyrics_vectorized = VECTORIZER.transform([input.lyrics])

    y_pred_binary = CLASSIFIER.predict(lyrics_vectorized)

    predicted_labels = MLB.inverse_transform(y_pred_binary)

    return{"predicted_genres": predicted_labels[0]}