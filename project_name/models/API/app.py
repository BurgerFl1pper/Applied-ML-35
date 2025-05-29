from fastapi import FastAPI
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
    lyrics: str


@app.post("/predict")
def predict_genres(input: LyricsInput):
    try:
        lyrics_vectorized = VECTORIZER.transform([input.lyrics])

        y_pred_binary = CLASSIFIER.predict(lyrics_vectorized)

        predicted_labels = MLB.inverse_transform(y_pred_binary)

        return{"predicted_genres": predicted_labels[0]}
    except Exception as e:
        return {"error": str(e)}