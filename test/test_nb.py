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
print("PREPARING INPUT: NAIVE BAYES\n")
print("####################################################################\n")

N_SAMPLES_PER_CLASS_TRAIN = 6500
N_SAMPLES_PER_CLASS_TEST = 500
N_SAMPLES_TRAIN = 146000
N_SAMPLES_TEST = 10000
PATH_TO_DATA = "../data/train.tsv"
CLASS_LABELS = [1, 2, 3, 4, 5]

train_documents, train_labels, train_end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TRAIN, 0, CLASS_LABELS, PATH_TO_DATA)
test_documents, test_labels, end_index = data_handler.load_balanced_rt_data(N_SAMPLES_PER_CLASS_TEST, train_end_index, CLASS_LABELS, PATH_TO_DATA)

#train_documents, train_labels, train_end_index = data_handler.load_rt_data(N_SAMPLES_TRAIN, 0, PATH_TO_DATA)
#test_documents, test_labels, end_index = data_handler.load_rt_data(N_SAMPLES_TEST, train_end_index, PATH_TO_DATA)

print("end_index", end_index)
train_input = data_handler.generate_nb_input(train_documents, train_labels, CLASS_LABELS)
test_input = data_handler.generate_nb_input(test_documents, test_labels, CLASS_LABELS)

print("#################################################################### \n")
print("TRAINING: NAIVE BAYES\n")
print("####################################################################\n")

nb_bernoulli_classifier = naive_bayes.NaiveBayesBernoulliClassifier()
nb_multinomial_classifier = naive_bayes.NaiveBayesMultinomialClassifier()
nb_multinomial_classifier.train(train_input)

#pickle.dump(nb_multinomial_classifier, open("../pickle/nb_multinomial_classifier.p", "wb"))

print("#################################################################### \n")
print("TESTING: NAIVE BAYES\n")
print("####################################################################\n")

#with open('../pickle/nb_multinomial_classifier.p', 'rb') as pickle_file:
#    nb_multinomial_classifier_from_file = pickle.load(pickle_file)

predictions, actual = nb_multinomial_classifier.test(test_input)
print("Predictions", predictions)
print("actual", actual)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
