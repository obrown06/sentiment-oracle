import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pre_process
import logistic_regression
import feature_extract

with open("train.ft.txt", 'r', encoding='utf8') as file:
    train_reviews = tuple(file)

with open("test.ft.txt", 'r', encoding='utf8') as file:
    test_reviews = tuple(file)

class_names = ["__label__1", "__label__2"]

print("#################################################################### \n")
print("TESTING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

print("Pre_processing...")

train_texts, train_labels = pre_process.strip_labels_and_clean(train_reviews[0:int(len(train_reviews) / 100)], class_names)
test_texts, test_labels = pre_process.strip_labels_and_clean(test_reviews[0:int(len(test_reviews) / 100)], class_names)

print("Extracting features...")

NFEATURES = 2000
NGRAMS = 2

extractor = feature_extract.FeatureExtractor()

print("len", len(train_texts))

feature_set = extractor.build_feature_set(train_texts, NFEATURES, NGRAMS)
train_input = extractor.extract_features(train_texts, feature_set, NGRAMS)
test_input = extractor.extract_features(test_texts, feature_set, NGRAMS)

print("train_input shape", train_input.shape)

print("Training...")

lr_classifier = logistic_regression.LogisticRegressionClassifier(NITERATIONS = 3000)
lr_classifier.train(train_input, train_labels, "batch")

print("Testing...\n")

POS_LABEL = 1
predictions, actual = lr_classifier.test(test_input, test_labels)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)


print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("AUC: ", auc)

print("####################################################################")
