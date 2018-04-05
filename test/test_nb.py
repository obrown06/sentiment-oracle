import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')

import pre_process
import naive_bayes
import math
import io
import sklearn
import test_utils
import pickle

def subset(documents, labels, label):
    subset = []
    for i in range(len(labels)):
        if labels[i] == label:
            subset.append(documents[i])

    return subset

with open("train.ft.txt", 'r', encoding='utf8') as file:
    train_reviews = tuple(file)

with open("test.ft.txt", 'r', encoding='utf8') as file:
    test_reviews = tuple(file)

class_names = ["__label__1", "__label__2"]

print("#################################################################### \n")
print("TESTING: NAIVE BAYES\n")
print("####################################################################\n")

print("Pre_processing...")

cleaner = pre_process.DocumentCleaner()

#train_documents, train_labels = cleaner.strip_labels_and_clean(train_reviews[0:int(len(train_reviews) // 10)], class_names)
test_documents, test_labels = cleaner.strip_labels_and_clean(test_reviews[0:int(len(test_reviews) // 10)], class_names)

train_data = dict()
test_data = dict()

for i in range(len(class_names)):
#    train_data[i] = subset(train_documents, train_labels, i)
    test_data[i] = subset(test_documents, test_labels, i)

nb_bernoulli_classifier = naive_bayes.NaiveBayesBernoulliClassifier()
nb_multinomial_classifier = naive_bayes.NaiveBayesMultinomialClassifier()

print("Training...")

#nb_bernoulli_classifier.train(train_data)
#nb_multinomial_classifier.train(train_data)

#pickle.dump(nb_bernoulli_classifier, open("../classifiers/nb_bernoulli_classifier.p", "wb"))
#pickle.dump(nb_multinomial_classifier, open("../classifiers/nb_multinomial_classifier.p", "wb"))

print("Testing...\n")

with open('../classifiers/nb_bernoulli_classifier.p', 'rb') as pickle_file:
    nb_bernoulli_classifier_from_file = pickle.load(pickle_file)

with open('../classifiers/nb_multinomial_classifier.p', 'rb') as pickle_file:
    nb_multinomial_classifier_from_file = pickle.load(pickle_file)

POS_LABEL = 1
predictions, actual = nb_bernoulli_classifier_from_file.test(test_data)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)

print("####################################################################\n")

print("RESULTS (Bernoulli): \n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("AUC: ", auc)

print("####################################################################")

POS_LABEL = 1
predictions, actual = nb_multinomial_classifier_from_file.test(test_data)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)

print("####################################################################\n")

print("RESULTS (Multinomial): \n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("AUC: ", auc)

print("####################################################################")
