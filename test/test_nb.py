import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')

import pre_process
import naive_bayes
import math
import io
import sklearn
import test_utils

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

train_documents, train_labels = pre_process.strip_labels_and_clean(train_reviews[0:int(len(train_reviews) // 100)], class_names)
test_documents, test_labels = pre_process.strip_labels_and_clean(test_reviews[0:int(len(test_reviews) // 100)], class_names)

train_data = dict()
test_data = dict()

for i in range(len(class_names)):
    train_data[i] = subset(train_documents, train_labels, i)
    test_data[i] = subset(test_documents, test_labels, i)

nb_classifier = naive_bayes.NaiveBayesBernoulliClassifier()

print("Training...")

nb_classifier.train(train_data)

print("Testing...\n")

POS_LABEL = 1
predictions, actual = nb_classifier.test(test_data)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)

print("####################################################################\n")
print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("AUC: ", auc)

print("####################################################################")
