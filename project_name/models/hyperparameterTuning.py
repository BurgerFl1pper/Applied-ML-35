import tensorflow as tf
import random
import numpy as np
from dataclasses import dict

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report


class HyperTuning:

    def __init__(self, run_amount, max_len, max_words, embedding_dim, mlb):
        self.run_amount = run_amount
        self.parameters = dict()
        self.best_parameters = self.parameters.copy()
        self.best_score = float('-inf')
        self.max_len = max_len
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.num_labels = len(mlb.classes_)
        self.neurons = [32, 64, 128, 256]
        self.density = [32, 64, 128]
        self.dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.alpha = [0.25, 0.5, 0.75]
        self.gamma = [1, 2, 5]
        self.batch_size = [32, 64, 128]
        self.learning_rate = [1e-4, 1e-3, 1e-2]

    def build(self,
              num_labels: int,
              layers: int,
              density: int,
              dropout: float,
              learning_rate: float):
        """
        Builds a model with the given hyperparameters
        :param num_labels: amount of output labels
        :param layers: hyperparameter to control the amount of hidden layers
        :param density: hyperparameter to control the density of each layer
        :param dropout: hyperparameter to control the dropout
        :param learning_rate: hyperparameter to control the learning rate
        :return: a built model
        """
        inputs = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.max_words, output_dim=self.embedding_dim,
                      input_length=self.max_len)(inputs)
        x = LSTM(layers, return_sequences=False)(x)
        x = Dense(density, activation='relu')(x)
        x = Dropout(dropout)(x)
        outputs = Dense(num_labels, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def chooseParameters(self):
        """
        Chooses random hyperparameters
        :return: list of hyperparameter values
        """
        neurons = random.choice(self.neurons)
        density = random.choice(self.density)
        dropout = random.choice(self.dropout)
        alpha = random.choice(self.alpha)
        gamma = random.choice(self.gamma)
        batch_size = random.choice(self.batch_size)
        learning_rate = random.choice(self.learning_rate)
        self.parameters = dict(lstmNeurons=neurons,
                               denseNeurons=density,
                               dropout=dropout,
                               alpha=alpha,
                               gamma=gamma,
                               batch_size=batch_size,
                               learning_rate=learning_rate)


    def fit(self, X_train_pad, X_val_pad,
            y_train_binary, y_val_binary):
        """
        Builds and trains a model on the input data using the learning rate
        hyperparameter. Uses early stopping.
        :param X_train_pad: input training data
        :param X_val_pad: input validation data
        :param y_train_binary: output training data as a binary
        :param y_val_binary: output validation data as a binary
        :return: trained model
        """
        # choose hyperparameters
        self.chooseParameters()

        # initiate and fit model
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    mode='min',
                                                    patience=1)

        model = self.build(self.num_labels,
                           self.parameters["lstmNeurons"],
                           self.parameters["denseNeurons"],
                           self.parameters["dropout"],
                           self.parameters["learning_rate"])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            Precision(name="precision"),
            Recall(name="recall")])

        model.fit(X_train_pad, y_train_binary,
                  epochs=20,
                  batch_size=self.parameters["batch_size"],
                  validation_data=(X_val_pad, y_val_binary),
                  callbacks=[callback])

        return model

    @staticmethod
    def make_binary(y_pred_prob, mlb):
        """
        Returns binary values for output based on their probabilities meeting
        a threshhold
        :param y_pred_prob: list of output prediction probabilities
        :return: list of output prediction binaries
        """
        y_pred_test = np.zeros_like(y_pred_prob)
        for i in range(len(mlb.classes_)):
            y_pred_test[:, i] = (y_pred_prob[:, i] >= 0.1).astype(int)
        return y_pred_test

    def evaluate(self, model, X_val_pad, y_val_binary, mlb):
        """
        Evaluate the model based on f1 score, and compares it to the current
        best mode
        :param model: model to be evaluated
        :param X_val_pad:
        :param y_val_binary:
        :param mlb:
        :return:
        """
        y_pred_prob_val = model.predict(X_val_pad)

        y_pred = self.make_binary(y_pred_prob_val, mlb)

        report = classification_report(y_val_binary, y_pred, target_names=mlb.classes_, output_dict=True)
        score = report["macro avg"]["f1-score"]
        if float(score) > self.best_score:
            self.best_score = score
            self.best_parameters = self.parameters.copy()

    def tuneParameters(self,
                       X_train_pad,
                       X_val_pad,
                       y_train_binary,
                       y_val_binary,
                       mlb):
        """
        Builds, trains and evaluates a chosen amount of models to tune
        hyperparameters.
        :param X_train_pad:
        :param X_val_pad:
        :param y_train_binary:
        :param y_val_binary:
        :param mlb:
        :return: List of best parameters
        """
        for i in range(self.run_amount):
            model = self.fit(X_train_pad,
                             X_val_pad,
                             y_train_binary,
                             y_val_binary)
            self.evaluate(model, X_val_pad, y_val_binary, mlb)

        return self.best_parameters
