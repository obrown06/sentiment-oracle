import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pickle
import json
import lstm_keras
import numpy as np
import feed_forward_pt
import data_handler
import keras

print("#################################################################### \n")
print("GENERATING INPUT: KERAS LSTM\n")
print("####################################################################\n")

AMAZON_PREFIX = "../pickle/amazon/"

PATH_TO_CLASSIFIER = AMAZON_PREFIX + "keras_lstm_classifier.h5"
PATH_TO_WRAPPER = AMAZON_PREFIX + "keras_lstm_wrapper.p"
PATH_TO_EXTRACTOR = AMAZON_PREFIX + "keras_lstm_extractor.p"

extractor = pickle.load(open(PATH_TO_EXTRACTOR, "rb"))
classifier = lstm_keras.load_keras(PATH_TO_CLASSIFIER, PATH_TO_WRAPPER)
data_info = classifier.data_info
classifier_info = classifier.classifier_info

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])

test_input = data_handler.generate_input(test_documents, extractor)
test_labels = np.array(test_labels)
test_label_input = keras.utils.to_categorical(test_labels - 1, len(data_info["class_labels"]))

print("#################################################################### \n")
print("DATA INFO: \n")
print("Source : ", data_info["source"])
print("Is distribution balanced? ", data_info["is_balanced"])
print("Number of training samples: ", data_info["n_samples_train"])
print("Number of validation samples: ", data_info["n_samples_val"])
print("Number of testing samples: ", data_info["n_samples_test"])
print("\n")
print("ARCHITECTURE INFO: \n")
print("embed_size: ", classifier_info["embed_size"])
print("vocab_size: ", classifier_info["vocab_size"])
print("embedding_output_dim: ", classifier_info["embedding_output_dim"])
print("lstm_dim: ", classifier_info["lstm_dim"])
print("dropout_ratio: ", classifier_info["dropout_ratio"])
print("\n")
print("TRAINING INFO: \n")
print("optimizer_type : ", classifier_info["optimizer_type"])
print("nepochs : ", classifier_info["nepochs"])
print("alpha: ", classifier_info["alpha"])
print("batch_size: ", classifier_info["batch_size"])

print("####################################################################\n")

predictions, actual = classifier.test(test_input, test_labels)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
