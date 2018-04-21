import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')

import naive_bayes
import data_handler
import math
import io
import sklearn
import test_utils
import pickle

print("#################################################################### \n")
print("TESTING: NAIVE BAYES\n")
print("####################################################################\n")

PATH_TO_CLASSIFIER = "../pickle/nb_multinomial_classifier.p"
classifier = pickle.load(open(PATH_TO_CLASSIFIER, "rb"))
data_info = classifier.data_info

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
test_input = data_handler.generate_nb_input(test_documents, test_labels, data_info["class_labels"])

print("#################################################################### \n")
print("DATA INFO:\n")
print("Source : ", data_info["source"])
print("Is distribution balanced? ", data_info["is_balanced"])
print("Number of training samples: ", data_info["n_samples_train"])
print("Number of testing samples: ", data_info["n_samples_test"])
print("####################################################################\n")

predictions, actual = classifier.test(test_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
