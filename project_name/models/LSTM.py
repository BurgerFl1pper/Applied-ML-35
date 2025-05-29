import os
import re
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from Random import RandomModel

max_words = 10000
max_len = 200
embedding_dim = 100


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


def lstm_model(num_labels):
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(inputs)
    x = LSTM(128, return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def predict(X_train_pad, X_test_pad, y_train_binary, y_test_binary, mlb):
    model = lstm_model(num_labels=len(mlb.classes_))
    model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=[
        Precision(name="precision"),
        Recall(name="recall")])

    model.fit(X_train_pad, y_train_binary, 
              epochs=1,
              batch_size=64, 
              validation_split=0.2)

    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)
    return y_pred
    
   
def main():
    finished_data = loadData()
    X_train, X_test, y_train, y_test = splitData(finished_data)
    
    X_train = cleanData(X_train)
    X_test = cleanData(X_test)
    
    X_train_pad, X_test_pad, y_train_binary, y_test_binary, tokenizer, mlb = tokenizingData(X_train, X_test, y_train, y_test)
    
    #y_pred = predict(X_train_pad, X_test_pad, y_train_binary, y_test_binary, mlb)

    print("RandomGuesses:")
    guesses = RandomModel(y_test_binary, mlb)

    #print(classification_report(y_test_binary, y_pred, target_names=mlb.classes_))


main()
