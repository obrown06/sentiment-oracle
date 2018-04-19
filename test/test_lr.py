import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')

import data_handler
import test_utils
import logistic_regression
import pickle
import numpy as np

print("#################################################################### \n")
print("GENERATING INPUT: LOGISTIC REGRESSION\n")
print("####################################################################\n")

N_SAMPLES_PER_CLASS_TRAIN = 20000
N_SAMPLES_PER_CLASS_TEST = 1000
N_SAMPLES_TRAIN = 130000
N_SAMPLES_TEST = 10000
NFEATURES = 2000
NGRAMS = 2
CLASS_LABELS = [1, 2, 3, 4, 5]
PATH_TO_DATA = "../data/train.tsv"

test_documents, test_labels, test_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TEST, 0, CLASS_LABELS, PATH_TO_DATA)
train_documents, train_labels, end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TRAIN, test_end_index, CLASS_LABELS, PATH_TO_DATA)
print("end_index: ", end_index)

extractor = data_handler.generate_bow_extractor(train_documents, NFEATURES, NGRAMS)
pickle.dump(extractor, open("../pickle/lr_extractor.p", "wb"))

train_input = data_handler.generate_bow_input(train_documents, extractor)
test_input = data_handler.generate_bow_input(test_documents, extractor)

train_label_input = np.array(train_labels)
test_label_input = np.array(test_labels)

print("#################################################################### \n")
print("TRAINING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

CLASS_LABELS = [1, 2, 3, 4, 5]
NITERATIONS = 1000
ALPHA = 0.1
LAMBDA = 1

lr_classifier = logistic_regression.LogisticRegressionClassifier(NITERATIONS, LAMBDA, ALPHA)
lr_classifier.train(train_input, train_label_input, CLASS_LABELS, "batch")
pickle.dump(lr_classifier, open("../pickle/lr_classifier.p", "wb"))

print("#################################################################### \n")
print("TESTING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

with open('../pickle/lr_classifier.p', 'rb') as pickle_file:
    lr_classifier_from_file = pickle.load(pickle_file)

predictions, actual = lr_classifier_from_file.test(test_input, test_label_input)
print("predictions", predictions[:100])
print("actual", actual[:100])
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
