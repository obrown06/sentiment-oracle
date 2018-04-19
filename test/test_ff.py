import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import data_handler
import feed_forward
import pickle
import numpy as np

print("#################################################################### \n")
print("GENERATING INPUT: FEED FORWARD\n")
print("####################################################################\n")

N_SAMPLES_PER_CLASS_TRAIN = 20000
N_SAMPLES_PER_CLASS_TEST = 1000
N_SAMPLES_TRAIN = 10000
N_SAMPLES_TEST = 1000
NFEATURES = 2000
NGRAMS = 2
PATH_TO_DATA = "../data/train.tsv"
CLASS_LABELS = [1, 2, 3, 4, 5]

test_documents, test_labels, test_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TEST, 0, CLASS_LABELS, PATH_TO_DATA)
train_documents, train_labels, end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TRAIN, test_end_index, CLASS_LABELS, PATH_TO_DATA)


#train_documents, train_labels, train_end_index = data_handler.load_data(N_SAMPLES_TRAIN, 0, PATH_TO_DATA)
#test_documents, test_labels, end_index = data_handler.load_data(N_SAMPLES_TEST, train_end_index, PATH_TO_DATA)

print("end_index: ", end_index)
extractor = data_handler.generate_bow_extractor(train_documents, NFEATURES, NGRAMS)
pickle.dump(extractor, open("../pickle/ff_extractor.p", "wb"))

train_input = data_handler.generate_bow_input(train_documents, extractor)
test_input = data_handler.generate_bow_input(test_documents, extractor)

train_label_input = np.array(train_labels)
test_label_input = np.array(test_labels)

print("#################################################################### \n")
print("TRAINING: FEED FORWARD\n")
print("#################################################################### \n")

NITERATIONS = 500
ALPHA = 0.001
LAMBDA = 0.5
layer_dims = [NFEATURES, 200, 5]

ff_classifier = feed_forward.FeedForwardClassifier(CLASS_LABELS, layer_dims, NITERATIONS, LAMBDA, ALPHA)
ff_classifier.train(train_input, train_label_input, "batch")

pickle.dump(ff_classifier, open("../pickle/ff_classifier.p", "wb"))

print("#################################################################### \n")
print("TESTING: FEED FORWARD\n")
print("#################################################################### \n")

with open('../pickle/ff_classifier.p', 'rb') as pickle_file:
    ff_classifier_from_file = pickle.load(pickle_file)

predictions, actual = ff_classifier_from_file.test(test_input, test_label_input)
print("predictions", predictions)
print("actual", actual)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
