import sys
sys.path.insert(0, '../test/')
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')

import naive_bayes
import data_handler
import math
import numpy as np
import io
import sklearn
import test_utils
import pickle

print("#################################################################### \n")
print("TESTING: PYTORCH LSTM\n")
print("####################################################################\n")
PATH_TO_EXTRACTOR = "../pickle/pytorch_lstm_extractor.p"
PATH_TO_CLASSIFIER = "../pickle/pytorch_lstm_classifier.p"
classifier = pickle.load(open(PATH_TO_CLASSIFIER, "rb"))
extractor = pickle.load(open(PATH_TO_EXTRACTOR, "rb"))

data_info = classifier.data_info
classifier_info = classifier.classifier_info

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
test_input = data_handler.generate_input(test_documents, extractor, SHOULD_ADD_NEGATIONS=False)
test_label_input = np.array(test_labels)

print("#################################################################### \n")
print("DATA INFO: \n")
print("Source : ", data_info["source"])
print("Is distribution balanced? ", data_info["is_balanced"])
print("Number of training samples: ", data_info["n_samples_train"])
print("Number of validation samples: ", data_info["n_samples_val"])
print("Number of testing samples: ", data_info["n_samples_test"])
print("\n")
print("CLASSIFIER INFO: \n")
print("nfeatures: ", classifier_info["nfeatures"])
print("embed_size: ", classifier_info["embed_size"])
print("hidden_dim: ", classifier_info["hidden_dim"])
print("\n")
print("TRAINING INFO: \n")
print("nepochs : ", classifier_info["nepochs"])
print("nbatches : ", classifier_info["nbatches"])
print("alpha: ", classifier_info["alpha"])
print("####################################################################\n")

predictions, actual = classifier.test(test_input, test_label_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
