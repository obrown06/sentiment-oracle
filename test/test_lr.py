import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pickle
import json
import numpy as np
import feed_forward_pt
import data_handler

print("#################################################################### \n")
print("TESTING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

YELP_PREFIX = "../pickle/yelp/"
AMAZON_PREFIX = "../pickle/amazon/"

PATH_TO_CLASSIFIER = YELP_PREFIX + "lr_classifier.p"
PATH_TO_EXTRACTOR = YELP_PREFIX + "lr_extractor.p"

classifier = pickle.load(open(PATH_TO_CLASSIFIER, "rb"))
extractor = pickle.load(open(PATH_TO_EXTRACTOR, "rb"))

data_info = classifier.data_info
classifier_info = classifier.classifier_info

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
test_input = data_handler.generate_input(test_documents, extractor)
test_label_input = np.array(test_labels)

print("#################################################################### \n")
print("DATA INFO: \n")
print("Source : ", data_info["source"])
print("Is distribution balanced? ", data_info["is_balanced"])
print("Number of training samples: ", data_info["n_samples_train"])
print("Number of validation samples: ", data_info["n_samples_val"])
print("Number of testing samples: ", data_info["n_samples_test"])
print("\n")
print("CLASSIFIER INFO: \n")
print("niterations : ", classifier_info["niterations"])
print("lambda: ", classifier_info["lambda"])
print("alpha: ", classifier_info["alpha"])
print("nfeatures: ", classifier_info["nfeatures"])
print("ngrams: ", classifier_info["ngrams"])
print("####################################################################\n")

predictions, actual = classifier.test(test_input, test_label_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)
precision, recall, specificity, accuracy, auc = test_utils.test_statistics(predictions, actual, pos_label=2)
print("####################################################################\n")

print("RESULTS: \n")
print("Accuracy: ", accuracy)
print("ONLY RELEVANT FOR FINE GRAINED:")
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)
print("ONLY RELEVANT FOR BINARY:")
print("Recall: ", recall)
print("Specificity: ", specificity)
print("Precision: ", precision)
print("AUC: ", auc)

print("####################################################################")
