import sys
sys.path.insert(0, '../test/')
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import data_handler
import test_utils
import logistic_regression
import pickle
import numpy as np

print("#################################################################### \n")
print("GENERATING INPUT: LOGISTIC REGRESSION\n")
print("####################################################################\n")

AMAZON_PREFIX = "../pickle/amazon/"
YELP_PREFIX = "../pickle/yelp/balanced/"

PATH_TO_CLASSIFIER = YELP_PREFIX + "lr_classifier.p"
PATH_TO_EXTRACTOR = YELP_PREFIX + "lr_extractor.p"

data_info = {"source" : "YELP",
             "path" : "../data/review.json",
             "is_balanced" : True,
             "n_samples_train" : 300000,
             "n_samples_val" : 10000,
             "n_samples_test" : 10000,
             "class_labels" : [1, 2, 3, 4, 5]
}

classifier_info = {"nfeatures" : 2000,
                   "ngrams" : 2,
                   "niterations" : 1000,
                   "alpha" : 0.1,
                   "lambda" : 1

}

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])

extractor = data_handler.generate_bag_of_ngrams_extractor(train_documents, classifier_info["nfeatures"], classifier_info["ngrams"])
pickle.dump(extractor, open(PATH_TO_EXTRACTOR, "wb"))

train_input = data_handler.generate_input(train_documents, extractor)
val_input = data_handler.generate_input(val_documents, extractor)

train_label_input = np.array(train_labels)
val_label_input = np.array(val_labels)

print("#################################################################### \n")
print("TRAINING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

lr_classifier = logistic_regression.LogisticRegressionClassifier(data_info, classifier_info)
lr_classifier.train(train_input, train_label_input, "batch")
pickle.dump(lr_classifier, open(PATH_TO_CLASSIFIER, "wb"))

print("#################################################################### \n")
print("VALIDATING: LOGISTIC REGRESSION\n")
print("####################################################################\n")

predictions, actual = lr_classifier.test(val_input, val_label_input)
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
