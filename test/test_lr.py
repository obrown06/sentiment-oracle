import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import pre_process
import test_utils
import logistic_regression
import bow_extractor
import pickle
import json
import numpy as np

print("#################################################################### \n")
print("LOADING FROM FILE: LOGISTIC REGRESSION\n")
print("####################################################################\n")

train_documents = []
train_labels = []

with open("review.json", 'r', encoding='utf8') as file:
    for i, line in enumerate(file):
        if i == 400:
            break
        data = json.loads(line)
        train_documents.append(data["text"])
        train_labels.append(data["stars"])

test_documents = train_documents[-40:]
test_labels = train_labels[-40:]
train_documents = train_documents[0:360]
train_labels = train_labels[0:360]

print("#################################################################### \n")
print("TRAINING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

print("Pre_processing...")

cleaner = pre_process.DocumentCleaner()

train_documents = cleaner.clean(train_documents[0:int(len(train_documents) // 1)])
test_documents = cleaner.clean(test_documents[0:int(len(test_documents) // 1)])

train_labels = train_labels[0:int(len(train_labels) // 1)]
test_labels = test_labels[0:int(len(test_labels) // 1)]

print("Extracting features...")

NFEATURES = 2000
NGRAMS = 2

extractor = bow_extractor.FeatureExtractor()
feature_set = extractor.build_feature_set(train_documents, NFEATURES, NGRAMS)

#with open('../data/lr_extractor.p', 'rb') as pickle_file:
#    extractor = pickle.load(pickle_file)

#feature_set = extractor.feature_set
#pickle.dump(extractor, open("../data/lr_extractor.p", "wb"))

train_input = extractor.extract_features(train_documents, feature_set)
test_input = extractor.extract_features(test_documents, feature_set)
class_list = [1, 2, 3, 4, 5]
train_label_input = np.array(train_labels)
test_label_input = np.array(test_labels)

print("Training...")

lr_classifier = logistic_regression.LogisticRegressionClassifier(NITERATIONS = 10000)
lr_classifier.train(train_input, train_label_input, class_list, "batch")

#pickle.dump(lr_classifier, open("../classifiers/lr_classifier.p", "wb"))

print("Testing...\n")

#with open('../classifiers/lr_classifier.p', 'rb') as pickle_file:
    #lr_classifier_from_file = pickle.load(pickle_file)

POS_LABEL = 1
predictions, actual = lr_classifier.test(test_input, test_label_input)
print("predictions", predictions)
print("actual", actual)
#precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
