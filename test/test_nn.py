import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pre_process
import deep_net
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

train_texts, train_labels = pre_process.strip_labels_and_clean(train_reviews[0:int(len(train_reviews) / 1000)], class_names)
test_texts, test_labels = pre_process.strip_labels_and_clean(test_reviews[0:int(len(train_reviews) / 1000)], class_names)

print("Extracting features...")

NFEATURES = 200
NGRAMS = 2
NITERATIONS = 2000
ALPHA = 0.5
LAMBDA = 1
layer_dims = [NFEATURES, 19, 5, 1]

extractor = feature_extract.FeatureExtractor()

feature_set = extractor.build_feature_set(train_texts, NFEATURES, NGRAMS)
train_input = extractor.extract_features(train_texts, feature_set)
test_input = extractor.extract_features(test_texts, feature_set)

print("Training...")

nn_classifier = deep_net.DeepNetClassifier(layer_dims, NITERATIONS, LAMBDA, ALPHA)
nn_classifier.train(train_input, train_labels, "stochastic")

print("Testing...\n")

POS_LABEL = 1
predictions, actual = nn_classifier.test(test_input, test_labels)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, POS_LABEL)


print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("AUC: ", auc)

print("####################################################################")
