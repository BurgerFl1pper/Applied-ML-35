from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pandas as pd
import os
import re

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

def splitData(finished_data: pd.DataFrame):
    """
    Splits the data into text and Genre, so it can be trained.
    :param finished_data: pandas dataframe of the data with column names "text"
    and "genre"
    :return:
    """
    X = finished_data['text'].tolist()
    y = finished_data['Genre'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=40)
    return X_train, X_test, y_train, y_test

def cleanText(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cleanData(data):
    return [cleanText(song) for song in data]
        
def tokenizingData(X_train, X_test, y_train, y_test):
    """
    Tokenizes the data into a format that can be used by the model.
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: encoded data
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    mlb = MultiLabelBinarizer()
    y_train_binary = mlb.fit_transform(y_train)
    y_test_binary = mlb.transform(y_test)

    return X_train_pad, X_test_pad, y_train_binary, y_test_binary, tokenizer, mlb

