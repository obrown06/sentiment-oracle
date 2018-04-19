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

EMBEDDING_OUTPUT_DIM = 32
MAX_WORD_COUNT = 60
LSTM_DIM = 50
DROPOUT_RATIO = 0.6

class KerasLSTMClassifier():

    def __init__(self, vocab_size, class_list):
        super(KerasLSTMClassifier, self).__init__()
        output_dim = len(class_list)
        self.class_list = class_list
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, EMBEDDING_OUTPUT_DIM, input_length=MAX_WORD_COUNT))
        self.model.add(LSTM(LSTM_DIM))
        self.model.add(Dropout(DROPOUT_RATIO))
        self.model.add(Dense(100, activation='relu', W_constraint=maxnorm(1)))
        self.model.add(Dense(20, activation='relu', W_constraint=maxnorm(1)))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.summary()


    def train(self, X_train, Y_train, X_val, Y_val, optimizer_type, ALPHA = 0.0001, EPOCHS = 60, BATCH_SIZE = 32):

        if optimizer_type == 'nadam':
            optimizer = keras.optimizers.Nadam(lr=ALPHA, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        else:
            optimizer = SGD(lr=ALPHA, nesterov=True, momentum=0.7, decay=1e-4)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

        self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        self.model.fit(X_train, Y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1,
                       validation_data=(X_val, Y_val), callbacks=[reduce_lr, early_stopping])

    def test(self, data, target):
        predictions = np.array([])

        for i in range(target.shape[0]):
            doc_data = data[i]
            predictions = np.append(predictions, self.classify(doc_data))

        return predictions, target

    def classify(self, data):
        class_index = self.predict(data)
        return self.class_list[class_index]

    def predict(self, data):
        data = np.array([data])
        predictions = self.model.predict(data)
        return np.argmax(predictions[0])

def pickle_keras(wrapper, path_to_keras, path_to_wrapper):
    wrapper.model.save(path_to_keras)
    wrapper.model = None
    pickle.dump(wrapper, open(path_to_wrapper, "wb"))

def load_keras(path_to_keras, path_to_wrapper):
    model = keras.models.load_model(path_to_keras)

    with open(path_to_wrapper, 'rb') as wrapper_file:
        wrapper = pickle.load(wrapper_file)

    wrapper.model = model

    return wrapper
