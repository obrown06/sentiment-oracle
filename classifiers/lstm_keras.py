import keras
import numpy as np
import math
import tensorflow as tf
import keras
import pickle
from keras.constraints import maxnorm
from keras.optimizers import SGD, Nadam
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding
from keras.layers.core import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.utils import np_utils

class KerasLSTMClassifier():

    def __init__(self, data_info, classifier_info):
        self.data_info = data_info
        self.classifier_info = classifier_info
        self.class_labels = data_info["class_labels"]
        self.alpha = classifier_info["alpha"]
        self.nepochs = classifier_info["nepochs"]
        self.batch_size = classifier_info["batch_size"]
        self.optimizer_type = classifier_info["optimizer_type"]

        self.model = Sequential()
        self.model.add(Embedding(classifier_info["vocab_size"], classifier_info["embedding_output_dim"], input_length=classifier_info["max_word_count"]))
        self.model.add(LSTM(classifier_info["lstm_dim"]))
        self.model.add(Dropout(classifier_info["dropout_ratio"]))
        self.model.add(Dense(100, activation='relu', W_constraint=maxnorm(1)))
        self.model.add(Dense(20, activation='relu', W_constraint=maxnorm(1)))
        self.model.add(Dense(len(self.class_labels), activation='softmax'))
        self.model.summary()


    def train(self, X_train, Y_train, X_val, Y_val):

        if self.optimizer_type == 'nadam':
            optimizer = keras.optimizers.Nadam(lr=self.alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        else:
            optimizer = SGD(lr=self.alpha, nesterov=True, momentum=0.7, decay=1e-4)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

        self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        self.model.fit(X_train, Y_train, epochs = self.nepochs, batch_size = self.batch_size, verbose = 1,
                       validation_data=(X_val, Y_val), callbacks=[reduce_lr, early_stopping])
        print(type(self.model))

    def test(self, data, target):
        predictions = np.array([])

        for i in range(target.shape[0]):
            doc_data = data[i]
            predictions = np.append(predictions, self.classify(doc_data))

        return predictions, target

    def classify(self, data):
        class_index = self.predict(data)
        return self.class_labels[class_index]

    def predict(self, data):
        data = np.array([data])
        predictions = self.model.predict(data)
        return np.argmax(predictions[0])

def pickle_keras(wrapper, path_to_keras, path_to_wrapper):
    tmp = wrapper.model
    wrapper.model.save(path_to_keras)
    wrapper.model = None
    pickle.dump(wrapper, open(path_to_wrapper, "wb"))
    wrapper.model = tmp

def load_keras(path_to_keras, path_to_wrapper):
    model = keras.models.load_model(path_to_keras)

    with open(path_to_wrapper, 'rb') as wrapper_file:
        wrapper = pickle.load(wrapper_file)

    wrapper.model = model

    return wrapper
