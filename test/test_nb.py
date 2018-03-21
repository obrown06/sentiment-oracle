import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import pre_process
import naive_bayes
import math
import io
import sklearn
from sklearn import metrics

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

print("###################################################################")
print("TESTING NAIVE BAYES")
print("####################################################################")
print("Beginning pre_processing.")

train_documents, train_labels = pre_process.pre_process(train_reviews[0:int(len(train_reviews) // 50)], class_names)
test_documents, test_labels = pre_process.pre_process(test_reviews, class_names)

print("Completed pre_processing.")

train_data = dict()
test_data = dict()

for i in range(len(class_names)):
    train_data[class_names[i]] = subset(train_documents, train_labels, i)
    test_data[class_names[i]] = subset(test_documents, test_labels, i)

nb_classifier = naive_bayes.NaiveBayesMultinomialClassifier()

print("Beginning training")
nb_classifier.train(train_data)
print("Completed training.")

print("Beginning testing.")
actual, predictions = nb_classifier.test(test_data)

print("Accuracy: ")
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label = 1)

print("AUC of predictions: {0}".format(metrics.auc(fpr, tpr)))
