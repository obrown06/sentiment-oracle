import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pre_process
import logistic_regression
import feature_extract
import pickle

with open("train.ft.txt", 'r', encoding='utf8') as file:
    train_reviews = tuple(file)

with open("test.ft.txt", 'r', encoding='utf8') as file:
    test_reviews = tuple(file)

class_names = ["__label__1", "__label__2"]

print("#################################################################### \n")
print("TESTING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

print("Pre_processing...")

cleaner = pre_process.DocumentCleaner()

train_texts, train_labels = cleaner.strip_labels_and_clean(train_reviews[0:int(len(train_reviews) / 10000)], class_names)
test_texts, test_labels = cleaner.strip_labels_and_clean(test_reviews[0:int(len(test_reviews) / 10000)], class_names)

print("Extracting features...")

NFEATURES = 2000
NGRAMS = 2

extractor = feature_extract.FeatureExtractor()

print("len", len(train_texts))

feature_set = extractor.build_feature_set(train_texts, NFEATURES, NGRAMS)

pickle.dump(extractor, open("../data/extractor.p", "wb"))

train_input = extractor.extract_features(train_texts, feature_set)
test_input = extractor.extract_features(test_texts, feature_set)

print("train_input shape", train_input.shape)

print("Training...")

lr_classifier = logistic_regression.LogisticRegressionClassifier(NITERATIONS = 3000)
lr_classifier.train(train_input, train_labels, "batch")

pickle.dump(lr_classifier, open("../classifiers/lr_classifier.p", "wb"))

print("Testing...\n")

with open('../classifiers/lr_classifier.p', 'rb') as pickle_file:
    lr_classifier_from_file = pickle.load(pickle_file)

POS_LABEL = 1
predictions, actual = lr_classifier_from_file.test(test_input, test_labels)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)


print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("AUC: ", auc)

print("####################################################################")
