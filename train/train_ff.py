import sys
sys.path.insert(0, '../test/')
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import data_handler
import feed_forward
import pickle
import numpy as np

print("#################################################################### \n")
print("GENERATING INPUT: FEED FORWARD\n")
print("####################################################################\n")

YELP_PREFIX = "../pickle/yelp/balanced/"
AMAZON_PREFIX = "../pickle/amazon/"
RT_PREFIX = "../pickle/rt/balanced/binary/"

PATH_TO_CLASSIFIER = RT_PREFIX + "ff_classifier.p"
PATH_TO_EXTRACTOR = RT_PREFIX + "ff_extractor.p"

data_info = {"source" : "ROTTEN_TOMATOES",
             "path" : "../data/train.tsv",
             "is_balanced" : True,
             "n_samples_train" : 6000,
             "n_samples_val" : 500,
             "n_samples_test" : 500,
             "class_labels" : [1, 2, 4, 5]
}

classifier_info = {
                   "method" : "batch",
                   "batch_size" : 500,
                   "nfeatures" : 2000,
                   "ngrams" : 2,
                   "niterations" : 600,
                   "alpha" : 0.01,
                   "lambda" : 1,
                   "layers_dims" : [2000, 200, 4]

}

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])
print("end_index", end_index)
extractor = data_handler.generate_bag_of_ngrams_extractor(train_documents, classifier_info["nfeatures"], classifier_info["ngrams"])
pickle.dump(extractor, open(PATH_TO_EXTRACTOR, "wb"))

train_input = data_handler.generate_input(train_documents, extractor)
val_input = data_handler.generate_input(val_documents, extractor)
test_input = data_handler.generate_input(test_documents, extractor)

train_label_input = np.array(train_labels)
val_label_input = np.array(val_labels)
test_label_input = np.array(test_labels)

print("#################################################################### \n")
print("TRAINING: FEED FORWARD\n")
print("#################################################################### \n")

ff_classifier = feed_forward.FeedForwardClassifier(data_info, classifier_info)
ff_classifier.train(train_input, train_label_input)
pickle.dump(ff_classifier, open(PATH_TO_CLASSIFIER, "wb"))

print("#################################################################### \n")
print("VALIDATING: FEED FORWARD\n")
print("#################################################################### \n")

predictions, actual = ff_classifier.test(val_input, val_label_input)
print("predictions", predictions)
print("actual", actual)
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
