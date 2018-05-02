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

data_info = {"source" : "YELP",
             "path" : "../data/review.json",
             "is_balanced" : True,
             "n_samples_train" : 300000,
             "n_samples_val" : 0,
             "n_samples_test" : 10000,
             "class_labels" : [1, 2, 3, 4, 5]}

AMAZON_PREFIX = "../pickle/amazon/"
YELP_PREFIX = "../pickle/yelp/balanced/"

PATH_TO_CLASSIFIER = YELP_PREFIX + "nb_multinomial_classifier.p"

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_train"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
print("end index: ", end_index)
train_input = data_handler.generate_nb_input(train_documents, train_labels, data_info["class_labels"])
test_input = data_handler.generate_nb_input(test_documents, test_labels, data_info["class_labels"])

print("#################################################################### \n")
print("TRAINING: NAIVE BAYES\n")
print("####################################################################\n")

nb_multinomial_classifier = naive_bayes.NaiveBayesMultinomialClassifier(data_info)
nb_multinomial_classifier.train(train_input)
pickle.dump(nb_multinomial_classifier, open(PATH_TO_CLASSIFIER, "wb"))

print("#################################################################### \n")
print("VALIDATING: NAIVE BAYES\n")
print("####################################################################\n")

predictions, actual = nb_multinomial_classifier.test(test_input)
print("Predictions: ", predictions[:100])
print("Actual: ", actual[:100])
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, pos_label=2)
print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("ONLY RELEVANT FOR FINE GRAINED:")
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)
print("ONLY RELEVANT FOR BINARY:")
print("Recall: ", recall)
print("Specificity: ", specificity)
print("Precision: ", precision)
print("AUC: ", auc)

print("####################################################################")
