from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os


class LyricsInput(BaseModel):
    text: str

async def lifespan(app: FastAPI):
    classifier, vectorizer, mlb = trainModel()
    app.state.classifier = classifier
    app.state.vectorizer = vectorizer
    app.state.mlb = mlb

    yield

app = FastAPI(lifespan=lifespan)

def loadData():
    """
    Finds the current file's directory, finds the data file directory relative
    to that and returns the data file
    :return:
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '..', 'data', 'finished_data.json')
    file_path = os.path.abspath(file_path)

    return pd.read_json(file_path, lines=True)

def trainModel():

    data = loadData()
    X = data['text'].tolist()
    y = data['Genre'].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(y)

    classifier = OneVsRestClassifier(LinearSVC())
    classifier.fit(X_vec, y_binary)

    return classifier, vectorizer, mlb

@app.post("/predict")
def predict_genres(request: Request, lyrics: LyricsInput):
    classifier = request.state.classifier
    vectorizer = request.state.vectorizer
    mlb = request.state.mlb

    X_vec = vectorizer.transform([lyrics.text])
    y_pred = classifier.predict(X_vec)
    predicted_labels = mlb.inverse_transform(y_pred)

    return{"predicted_genres": predicted_labels[0]}