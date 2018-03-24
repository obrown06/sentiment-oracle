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

train_texts, train_labels = pre_process.pre_process(train_reviews[0:int(len(train_reviews) / 1000)], class_names)
test_texts, test_labels = pre_process.pre_process(test_reviews[0:int(len(train_reviews) / 1000)], class_names)

print("Extracting features...")

NFEATURES = 2000
NGRAMS = 2
NITERATIONS = 2000
ALPHA = 0.5
LAMBDA = 1
layer_dims = [NFEATURES, 19, 5, 1]

feature_set = feature_extract.build_feature_set(train_texts, NFEATURES, NGRAMS)
train_input = feature_extract.input_matrix(train_texts, feature_set, NGRAMS)
test_input = feature_extract.input_matrix(test_texts, feature_set, NGRAMS)

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
