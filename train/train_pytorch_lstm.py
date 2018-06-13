import sys
sys.path.insert(0, '../test/')
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import glove_extractor
import lstm_pytorch
import pickle
import data_handler
import numpy as np

print("#################################################################### \n")
print("GENERATING INPUT : LSTM\n")
print("####################################################################\n")

# (Unbalanced) n_samples_train: 124000
# (Unbalanced) n_samples_val: 1000
# (Unbalanced) n_samples_test: 1000

YELP_PREFIX = "../pickle/yelp/balanced/"
AMAZON_PREFIX = "../pickle/amazon/balanced/"

data_info = {"source" : "ROTTEN_TOMATOES",
             "path" : "../data/train.tsv",
             "is_balanced" : True,
             "n_samples_train" : 500,
             "n_samples_val" : 100,
             "n_samples_test" : 100,
             "class_labels" : [1, 2, 3, 4, 5]
}

classifier_info = {"embed_size" : 300,
                   "nfeatures" : 1000,
                   "hidden_dim" : 40,
                   "nbatches" : 100,
                   "nepochs" : 1,
                   "alpha" : 0.001
}

PATH_TO_GLOVE_EMBEDDINGS = '../data/glove.42B.300d.txt'
PATH_TO_EXTRACTOR = "../pickle/pytorch_lstm_extractor.p"
train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])

extractor = data_handler.generate_glove_extractor(train_documents, classifier_info["nfeatures"])
embeddings = data_handler.generate_glove_embeddings(extractor, PATH_TO_GLOVE_EMBEDDINGS, classifier_info["nfeatures"], classifier_info["embed_size"])
pickle.dump(extractor, open(PATH_TO_EXTRACTOR, "wb"))

train_input = data_handler.generate_input(train_documents, extractor, SHOULD_ADD_NEGATIONS=False)
val_input = data_handler.generate_input(val_documents, extractor, SHOULD_ADD_NEGATIONS=False)

train_label_input = np.array(train_labels)
val_label_input = np.array(val_labels)

train_label_class_indices = data_handler.labels_to_indices(train_label_input, data_info["class_labels"])

print("#################################################################### \n")
print("TRAINING: LSTM\n")
print("#################################################################### \n")

lstm_classifier = lstm_pytorch.PyTorchLSTMClassifier(data_info, classifier_info)
lstm_classifier.train(train_input, train_label_class_indices, embeddings)
pickle.dump(lstm_classifier, open("../pickle/pytorch_lstm_classifier.p", "wb"))

print("#################################################################### \n")
print("VALIDATING: LSTM\n")
print("#################################################################### \n")

predictions, actual = lstm_classifier.test(val_input, val_label_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
