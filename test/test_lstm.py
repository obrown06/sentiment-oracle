import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import glove_extractor
import lstm
import pickle
import data_handler
import numpy as np

print("#################################################################### \n")
print("GENERATING INPUT : LSTM\n")
print("####################################################################\n")

N_SAMPLES_PER_CLASS_TRAIN = 10000
N_SAMPLES_PER_CLASS_TEST = 1000
NFEATURES = 2000
EMBED_SIZE = 300
NGRAMS = 2
CLASS_LABELS = [1, 2, 3, 4, 5]
PATH_TO_DATA = "../data/review.json"
PATH_TO_GLOVE_EMBEDDINGS = '../data/glove.42B.300d.txt'

train_documents, train_labels, train_end_index = data_handler.load_balanced_data(N_SAMPLES_PER_CLASS_TRAIN, 0, CLASS_LABELS, PATH_TO_DATA)
test_documents, test_labels, end_index = data_handler.load_balanced_data(N_SAMPLES_PER_CLASS_TEST, train_end_index, CLASS_LABELS, PATH_TO_DATA)
print("end_index: ", end_index)
extractor = data_handler.generate_glove_extractor(train_documents, NFEATURES, NGRAMS)
embeddings = data_handler.generate_glove_embeddings(extractor, PATH_TO_GLOVE_EMBEDDINGS, NFEATURES, EMBED_SIZE)
pickle.dump(extractor, open("../pickle/pytorch_lstm_extractor.p", "wb"))

train_input = data_handler.generate_glove_input(train_documents, extractor)
test_input = data_handler.generate_glove_input(test_documents, extractor)

train_label_input = np.array(train_labels)
test_label_input = np.array(test_labels)

train_label_class_indices = data_handler.labels_to_indices(train_label_input, CLASS_LABELS)

print("#################################################################### \n")
print("TRAINING: LSTM\n")
print("#################################################################### \n")

HIDDEN_DIM = 40
NBATCHES = 100
NEPOCHS = 10
ALPHA = 0.001

lstm_classifier = lstm.LSTMClassifier(EMBED_SIZE, HIDDEN_DIM, NFEATURES, CLASS_LABELS, embeddings)
lstm_classifier.train(train_input, train_label_class_indices, ALPHA, NEPOCHS, NBATCHES)
pickle.dump(lstm_classifier, open("../pickle/lstm_classifier.p", "wb"))

print("#################################################################### \n")
print("TESTING: LSTM\n")
print("#################################################################### \n")

predictions, actual = lstm_classifier.test(test_input, test_label_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
