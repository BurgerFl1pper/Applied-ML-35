from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import joblib


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


def encodingData(X_train, X_test, y_train, y_test):
    """
    Encodes the data into a format that can be used by the model.
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: encoded data
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    mlb = MultiLabelBinarizer()
    y_train_binary = mlb.fit_transform(y_train)
    y_test_binary = mlb.transform(y_test)

    return X_train_vec, X_test_vec, y_train_binary, y_test_binary, vectorizer, mlb

def trainClassifier(X_train_vec, y_train_binary):
    """ 
    Trains a OneVsRestClassifier with LinearSVC on the training data.
    :param X_train_vec: training data
    :param y_train_binary: training labels
    :return: trained classifier
    """
    clf = OneVsRestClassifier(LinearSVC())
    clf.fit(X_train_vec, y_train_binary)
    return clf

def evaluate(clf, X_test_vec, y_test_binary, mlb):
    """ 
    Evaluates the classifier on test data and prints a classification report.
    :param clf: trained classifier
    :param X_test_vec: testing data
    :param y_test_binary: testing labels
    :param mlb: MultiLabelBinarizer instance (for target class names)
    :return: None
    """
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test_binary, y_pred, target_names=mlb.classes_))

def saveAll(vectorizer, mlb, clf, model_dir='API'):
    """ 
    Saves the vectorizer, label binarizer, and classifier to disk.
    :param vectorizer: trained vectorizer
    :param mlb: trained MultiLabelBinarizer
    :param clf: trained classifier
    :param model_dir: directory path to save the models
    :return: None
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
    joblib.dump(mlb, os.path.join(model_dir, 'mlb.joblib'))
    joblib.dump(clf, os.path.join(model_dir, 'classifier.joblib'))

def main():
    finished_data = loadData()
    X_train, X_test, y_train, y_test = splitData(finished_data)
    X_train_vec, X_test_vec, y_train_binary, y_test_binary, vectorizer, mlb = encodingData(X_train,
                                                                          X_test,
                                                                          y_train,
                                                                          y_test)
    clf = trainClassifier(X_train_vec, y_train_binary)
    evaluate(clf, X_test_vec, y_test_binary, mlb)
    saveAll(vectorizer, mlb, clf)
    
main()