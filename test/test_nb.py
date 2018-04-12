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
import json

def subset(documents, labels, label):
    subset = []
    for i in range(len(labels)):
        if labels[i] == label:
            subset.append(documents[i])

    return subset

print("#################################################################### \n")
print("LOADING FROM FILE: NAIVE BAYES\n")
print("####################################################################\n")

train_documents = []
train_labels = []

with open("review.json", 'r', encoding='utf8') as file:
    for i, line in enumerate(file):
        if i == 2000000:
            break
        data = json.loads(line)
        train_documents.append(data["text"])
        train_labels.append(data["stars"])

test_documents = train_documents[-400000:]
test_labels = train_labels[-400000:]
train_documents = train_documents[0:3600000]
train_labels = train_labels[0:3600000]

class_names = [1, 2, 3, 4, 5]

print("#################################################################### \n")
print("TESTING: NAIVE BAYES\n")
print("####################################################################\n")

print("Pre_processing...")

cleaner = pre_process.DocumentCleaner()

train_documents = cleaner.clean(train_documents[0:int(len(train_documents) // 1000)])
test_documents = cleaner.clean(test_documents[0:int(len(test_documents) // 1000)])

train_labels = train_labels[0:int(len(train_labels) // 1000)]
test_labels = test_labels[0:int(len(test_labels) // 1000)]

train_data = dict()
test_data = dict()

for i in range(len(class_names)):
    train_data[i] = subset(train_documents, train_labels, class_names[i])
    test_data[i] = subset(test_documents, test_labels, class_names[i])

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
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS (Bernoulli): \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)
#print("Precision: ", precision)
#print("Recall: ", recall)
#print("Specificity: ", specificity)
#print("AUC: ", auc)

print("####################################################################")

POS_LABEL = 1
predictions, actual = nb_multinomial_classifier_from_file.test(test_data)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS (Multinomial): \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

#print("Precision: ", precision)
#print("Recall: ", recall)
#print("Specificity: ", specificity)
#print("AUC: ", auc)

print("####################################################################")
