import os
import re

import tensorflow.keras.callbacks
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

from focal_loss import MyBinaryFocalCrossentropy
from hyperparameterTuning import HyperTuning

import matplotlib.pyplot as plt
import numpy as np

max_words = 10000
max_len = 200
embedding_dim = 100


def loadData():
    """
    Loads the finished dataset from the JSON file.
    :return: pandas DataFrame containing the data
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '..', 'data', 'finished_data.json')
    file_path = os.path.abspath(file_path)

    return pd.read_json(file_path, lines=True)


def splitData(finished_data: pd.DataFrame):
    """
    Splits the dataset into training, validation, and test sets.
    :param finished_data: DataFrame with columns 'text' and 'Genre'
    :return: Tuple containing train/val/test splits for X and y
    """
    X = finished_data['text'].tolist()
    y = finished_data['Genre'].tolist()

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=40)

    return X_train, X_val, X_test, y_train, y_val, y_test


def cleanText(text):
    """
    Cleans a single text string by removing brackets, punctuation, and making it lowercase.
    :param text: Raw text string
    :return: Cleaned text string
    """
    text = re.sub(r"\[.*?\]", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cleanData(data):
    """
    Cleans a list of text data.
    :param data: List of raw text strings
    :return: List of cleaned text strings
    """
    return [cleanText(song) for song in data]


def tokenizingData(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tokenizes and pads text data and binarizes genre labels.
    :param X_train: List of training text
    :param X_val: List of validation text
    :param X_test: List of test text
    :param y_train: Training genre labels
    :param y_val: Validation genre labels
    :param y_test: Test genre labels
    :return: Tuple of tokenized and padded text data and binarized labels, tokenizer, and label binarizer
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    mlb = MultiLabelBinarizer()
    y_train_binary = mlb.fit_transform(y_train)
    y_val_binary = mlb.transform(y_val)
    y_test_binary = mlb.transform(y_test)

    return X_train_pad, X_val_pad, X_test_pad, y_train_binary, y_val_binary, y_test_binary, tokenizer, mlb


def lstmModel(num_labels, lstmNeurons=128, denseNeurons=64, dropout=0.1, learning_rate=1e-4, alpha=0.25, gamma=0.5):
    """
    Builds and compiles an LSTM-based multi-label classification model.
    :param num_labels: Number of output labels
    :param lstmNeurons: Number of LSTM neurons
    :param denseNeurons: Number of dense layer neurons
    :param dropout: Dropout rate
    :param learning_rate: Learning rate
    :param alpha: Focal loss alpha parameter
    :param gamma: Focal loss gamma parameter
    :return: Compiled Keras model
    """
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(inputs)
    x = LSTM(lstmNeurons, return_sequences=False)(x)
    x = Dense(denseNeurons, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    loss = MyBinaryFocalCrossentropy(alpha=alpha, gamma=gamma)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def computeClassThreshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Computes the optimal threshold for a single class based on F1 score.
    :param y_true: Ground truth binary labels for a class
    :param y_pred_prob: Predicted probabilities for a class
    :return: Optimal threshold
    """
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def computeOptimalThresholds(y_val_binary: np.ndarray, y_pred_prob_val: np.ndarray) -> np.ndarray:
    """
    Computes optimal thresholds for each class using validation set.
    :param y_val_binary: Ground truth binary labels for validation set
    :param y_pred_prob_val: Predicted probabilities for validation set
    :return: Array of optimal thresholds per class
    """
    thresholds = []
    for i in range(y_val_binary.shape[1]):
        threshold = computeClassThreshold(y_val_binary[:, i], y_pred_prob_val[:, i])
        thresholds.append(threshold)
    return np.array(thresholds)


def predict(X_train_pad, X_val_pad, X_test_pad, 
            y_train_binary, y_val_binary, y_test_binary, mlb, hyperparameters):
    """
    Trains the model, evaluates it on validation set, computes thresholds, and predicts on test set.
    :param X_train_pad: Padded training data
    :param X_val_pad: Padded validation data
    :param X_test_pad: Padded test data
    :param y_train_binary: Binarized training labels
    :param y_val_binary: Binarized validation labels
    :param y_test_binary: Binarized test labels
    :param mlb: MultiLabelBinarizer instance
    :param hyperparameters: Dictionary of tuned hyperparameters
    :return: Predicted probabilities, binary predictions, and optimal thresholds
    """
    callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        mode='min',
                                                        patience=1)
    model = lstmModel(num_labels=len(mlb.classes_),
                      lstmNeurons=hyperparameters["lstmNeurons"],
                      denseNeurons=hyperparameters["denseNeurons"],
                      dropout=hyperparameters["dropout"],
                      learning_rate=hyperparameters["learning_rate"])

    history = model.fit(X_train_pad, y_train_binary,
                        epochs=20,
                        batch_size=hyperparameters["batch_size"],
                        validation_data=(X_val_pad, y_val_binary),
                        callbacks=[callback])
    print("Epochs trained: ", len(history.history["val_loss"]))

    plot_training_history(history)

    y_pred_prob_val = model.predict(X_val_pad)
    thresholds = computeOptimalThresholds(y_val_binary, y_pred_prob_val)
    
    y_pred_prob_test = model.predict(X_test_pad)
    
    y_pred_test = np.zeros_like(y_pred_prob_test)
    for i, t in enumerate(thresholds):
        y_pred_test[:, i] = (y_pred_prob_test[:, i] >= t).astype(int)
    
    return y_pred_prob_test, y_pred_test, thresholds


def zeroRuleBaseline(y_true: np.ndarray, class_names: list):
    """
    Implements a zero-rule baseline by predicting all genres for every sample.
    :param y_true: Ground truth labels
    :param class_names: List of genre names
    :return: None
    """
    y_pred = np.ones(y_true.shape)
    print("Zero Rule Baseline Predictions:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plotPrecisionRecall(y_test_binary, y_pred_prob, genres):
    """
    Plots the Precision-Recall curve for each genre.
    :param y_test_binary: Ground truth binary labels
    :param y_pred_prob: Predicted probabilities
    :param genres: List of genre names
    :return: None
    """
    plt.figure(figsize=(10, 8))
    n_classes = len(genres)
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_binary[:, i], y_pred_prob[:, i])
        average_precision = average_precision_score(y_test_binary[:, i], y_pred_prob[:, i])
        plt.plot(recall, precision, lw=2, color=colors(i),
                 label=f"{genres[i]} (Average Precision={average_precision:.2f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve per Genre')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_training_history(history):
    """
    Plots training and validation loss curves.
    :param history: Keras training history object
    :return: None
    """
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label="Training Loss", color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label="Validation Loss", color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the full training, evaluation, and visualization pipeline.
    :return: None
    """
    finished_data = loadData()
    X_train, X_val, X_test, y_train, y_val, y_test = splitData(finished_data)
    
    X_train = cleanData(X_train)
    X_val = cleanData(X_val)
    X_test = cleanData(X_test)
    
    X_train_pad, X_val_pad, X_test_pad, y_train_binary, y_val_binary, y_test_binary, tokenizer, mlb = tokenizingData(
        X_train, X_val, X_test, y_train, y_val, y_test)

    tuner = HyperTuning(50,
                        max_len,
                        max_words,
                        embedding_dim, mlb)

    hyperparameters = tuner.tuneParameters(X_train_pad,
                                           X_val_pad,
                                           y_train_binary,
                                           y_val_binary,
                                           mlb)
    print(hyperparameters)

    y_pred_prob, y_pred, thresholds = predict(
        X_train_pad, X_val_pad, X_test_pad, y_train_binary, y_val_binary, y_test_binary, mlb, hyperparameters)
    
    print("Optimal thresholds per class:")
    for genre, threshold in zip(mlb.classes_, thresholds):
        print(f"{genre}: {threshold:.2f}")

    zeroRuleBaseline(y_test_binary, mlb.classes_)

    print("Our Model:")
    print(classification_report(y_test_binary, y_pred, target_names=mlb.classes_))

    plotPrecisionRecall(y_test_binary, y_pred_prob, mlb.classes_)


if __name__ == "__main__":
    main()