import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
sys.path.insert(0, '../test/')

import naive_bayes
import data_handler
import math
import io
import sklearn
import test_utils
import pickle

print("#################################################################### \n")
print("PREPARING INPUT: NAIVE BAYES\n")
print("####################################################################\n")

# (unbalanced) n_samples_train: 146000
# (unbalanced) n_samples_test: 10000

data_info = {"source" : "ROTTEN_TOMATOES",
             "path" : "../data/train.tsv",
             "is_balanced" : False,
             "n_samples_train" : 146000,
             "n_samples_val" : 0,
             "n_samples_test" : 10000,
             "class_labels" : [1, 2, 3, 4, 5]}

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_train"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
train_input = data_handler.generate_nb_input(train_documents, train_labels, data_info["class_labels"])
test_input = data_handler.generate_nb_input(test_documents, test_labels, data_info["class_labels"])

print("#################################################################### \n")
print("TRAINING: NAIVE BAYES\n")
print("####################################################################\n")

nb_multinomial_classifier = naive_bayes.NaiveBayesMultinomialClassifier(data_info)
nb_multinomial_classifier.train(train_input)
pickle.dump(nb_multinomial_classifier, open("../pickle/nb_multinomial_classifier.p", "wb"))

print("#################################################################### \n")
print("TESTING: NAIVE BAYES\n")
print("####################################################################\n")

predictions, actual = nb_multinomial_classifier.test(test_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
