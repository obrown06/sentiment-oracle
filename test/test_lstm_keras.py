import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import keras
import pickle
import keras_extractor
import lstm_keras
import data_handler
import numpy as np

def shuffle_2(a, b): # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]

print("#################################################################### \n")
print("GENERATING INPUT : LSTM\n")
print("####################################################################\n")

N_SAMPLES_PER_CLASS_TRAIN = 20000
N_SAMPLES_PER_CLASS_VAL = 1000
N_SAMPLES_PER_CLASS_TEST = 1000
N_SAMPLES_TRAIN = 124000
N_SAMPLES_VAL = 22000 + 1
N_SAMPLES_TEST = 10000
NFEATURES = 2000
EMBED_SIZE = 300
NGRAMS = 2
CLASS_LABELS = [1, 2, 3, 4, 5]
NCLASSES = 5
PATH_TO_DATA = "../data/train.tsv"
PATH_TO_GLOVE_EMBEDDINGS = '../data/glove.42B.300d.txt'
PATH_TO_WRAPPER_FILE = "../pickle/lstm_wrapper.p"
PATH_TO_KERAS_FILE = "../pickle/lstm_keras.h5"

val_documents, val_labels, val_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_VAL, 0, CLASS_LABELS, PATH_TO_DATA)
test_documents, test_labels, test_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TEST, val_end_index, CLASS_LABELS, PATH_TO_DATA)
train_documents, train_labels, train_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TRAIN, test_end_index, CLASS_LABELS, PATH_TO_DATA)

print("train_end_index: ", train_end_index)

extractor = data_handler.generate_keras_extractor(np.array(train_documents))
pickle.dump(extractor, open("../pickle/keras_lstm_extractor.p", "wb"))

train_input = data_handler.generate_keras_input(train_documents, extractor)
val_input = data_handler.generate_keras_input(val_documents, extractor)
test_input = data_handler.generate_keras_input(test_documents, extractor)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

train_label_input = keras.utils.to_categorical(train_labels - 1, NCLASSES)
val_label_input = keras.utils.to_categorical(val_labels - 1, NCLASSES)
test_label_input = keras.utils.to_categorical(test_labels - 1, NCLASSES)

print("#################################################################### \n")
print("TRAINING: LSTM\n")
print("#################################################################### \n")

NBATCHES = 100
BATCH_SIZE = 32
NEPOCHS = 10
ALPHA = 0.002

vocab_size = extractor.vocab_size()
lstm_classifier = lstm_keras.KerasLSTMClassifier(vocab_size, CLASS_LABELS)
shuffle_2(train_input, train_label_input)

lstm_classifier.train(train_input, train_label_input, val_input, val_label_input, "nadam", ALPHA, NEPOCHS, BATCH_SIZE)
lstm_keras.pickle_keras(lstm_classifier, PATH_TO_KERAS_FILE, PATH_TO_WRAPPER_FILE)

print("#################################################################### \n")
print("TESTING: LSTM\n")
print("#################################################################### \n")

lstm_classifier = lstm_keras.load_keras(PATH_TO_KERAS_FILE, PATH_TO_WRAPPER_FILE)
predictions, actual = lstm_classifier.test(test_input, test_labels)
print("predictions", predictions[:500])
print("actual", actual[:500])
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
