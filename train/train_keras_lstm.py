import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../test/')
sys.path.insert(0, '../classifiers/')
import keras
import pickle
import test_utils
import keras_extractor
import lstm_keras
import data_handler
import numpy as np

def shuffle(a, b):
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]

print("#################################################################### \n")
print("GENERATING INPUT : LSTM\n")
print("####################################################################\n")

AMAZON_PREFIX = "../pickle/amazon/"
YELP_PREFIX = "../pickle/yelp/balanced/"

PATH_TO_CLASSIFIER = YELP_PREFIX + "keras_lstm_classifier.h5"
PATH_TO_WRAPPER = YELP_PREFIX + "keras_lstm_wrapper.p"
PATH_TO_EXTRACTOR = YELP_PREFIX + "keras_lstm_extractor.p"

data_info = {"source" : "YELP",
             "path" : "../data/review.json",
             "is_balanced" : True,
             "n_samples_train" : 200000,
             "n_samples_val" : 20000,
             "n_samples_test" : 20000,
             "class_labels" : [1, 2, 3, 4, 5]
}

classifier_info = {"embed_size" : 300,
                   "batch_size" : 64,
                   "nepochs" : 3,
                   "alpha" : 0.001,
                   "optimizer_type" : "nadam",
                   "embedding_output_dim" : 32,
                   "max_word_count" : 60,
                   "lstm_dim" : 50,
                   "dropout_ratio" : 0.6
}


train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
extractor = data_handler.generate_keras_extractor(np.array(train_documents))
pickle.dump(extractor, open(PATH_TO_EXTRACTOR, "wb"))

train_input = data_handler.generate_input(train_documents, extractor)
val_input = data_handler.generate_input(val_documents, extractor)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

train_label_input = keras.utils.to_categorical(train_labels - 1, len(data_info["class_labels"]))
val_label_input = keras.utils.to_categorical(val_labels - 1, len(data_info["class_labels"]))

print("#################################################################### \n")
print("TRAINING: LSTM\n")
print("#################################################################### \n")

vocab_size = extractor.vocab_size()
classifier_info["vocab_size"] = vocab_size

lstm_classifier = lstm_keras.KerasLSTMClassifier(data_info, classifier_info)
shuffle(train_input, train_label_input)

lstm_classifier.train(train_input, train_label_input, val_input, val_label_input)
lstm_keras.pickle_keras(lstm_classifier, PATH_TO_CLASSIFIER, PATH_TO_WRAPPER)

print("#################################################################### \n")
print("VALIDATING: LSTM\n")
print("#################################################################### \n")

predictions, actual = lstm_classifier.test(val_input, val_labels)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
