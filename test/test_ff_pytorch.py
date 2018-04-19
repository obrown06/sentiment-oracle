import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import bow_extractor
import pickle
import json
import numpy as np
import feed_forward_pt
import data_handler

print("#################################################################### \n")
print("GENERATING INPUT: PYTORCH FEED FORWARD\n")
print("####################################################################\n")

N_SAMPLES_PER_CLASS_TRAIN = 9000
N_SAMPLES_PER_CLASS_TEST = 500
N_SAMPLES_TRAIN = 145000
N_SAMPLES_TEST = 10000
NFEATURES = 2000
NGRAMS = 2
CLASS_LABELS = [1, 2, 3, 4, 5]
PATH_TO_DATA = "../data/train.tsv"

test_documents, test_labels, test_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TEST, 0, CLASS_LABELS, PATH_TO_DATA)
train_documents, train_labels, end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TRAIN, test_end_index, CLASS_LABELS, PATH_TO_DATA)
print("end_index:", end_index)

extractor = data_handler.generate_bow_extractor(train_documents, NFEATURES, NGRAMS)
pickle.dump(extractor, open("../pickle/pytorch_ff_extractor.p", "wb"))

train_input = data_handler.generate_bow_input(train_documents, extractor)
test_input = data_handler.generate_bow_input(test_documents, extractor)

train_label_input = np.array(train_labels)
test_label_input = np.array(test_labels)

train_label_class_indices = data_handler.labels_to_indices(train_label_input, CLASS_LABELS)

print("#################################################################### \n")
print("TRAINING: PYTORCH FEED FORWARD\n")
print("#################################################################### \n")

# Alpha = 0.001 and NEPOCHS = 200 and NBATCHES = 50 and optim = SGD gives 0.675 accuracy, 0.8515 polarity, 0.93 near accuracy

NEPOCHS = 500
ALPHA = 0.001
NBATCHES = 50
INPUT_DIM = NFEATURES
HIDDEN_DIM = 200
OUTPUT_DIM = 5

pytorch_ff_classifier = feed_forward_pt.FeedForwardClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, CLASS_LABELS)
pytorch_ff_classifier.train(train_input, train_label_class_indices, ALPHA, NEPOCHS, NBATCHES)
pickle.dump(pytorch_ff_classifier, open("../pickle/pytorch_ff_classifier.p", "wb"))

print("#################################################################### \n")
print("TESTING: PYTORCH FEED FORWARD\n")
print("#################################################################### \n")

with open('../pickle/pytorch_ff_classifier.p', 'rb') as pickle_file:
    pytorch_ff_classifier = pickle.load(pickle_file)

predictions, actual = pytorch_ff_classifier.test(test_input, test_label_input)
print("predictions", predictions[:100])
print("actual", actual[:100])
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
