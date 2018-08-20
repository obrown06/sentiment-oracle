import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
sys.path.insert(0, '../test/')
import test_utils
import pickle
import json
import numpy as np
import feed_forward_pt
import data_handler

print("#################################################################### \n")
print("GENERATING INPUT: PYTORCH FEED FORWARD\n")
print("####################################################################\n")

YELP_PREFIX = "../pickle/yelp/balanced/"
AMAZON_PREFIX = "../pickle/amazon/balanced/"
RT_PREFIX = "../pickle/rt/balanced/binary/"

PATH_TO_CLASSIFIER = YELP_PREFIX + "pytorch_ff_classifier.p"
PATH_TO_EXTRACTOR = YELP_PREFIX + "pytorch_ff_extractor.p"


data_info = {"source" : "YELP",
             "path" : "../data/review.json",
             "is_balanced" : True,
             "n_samples_train" : 200000,
             "n_samples_val" : 10000,
             "n_samples_test" : 10000,
             "class_labels" : [1, 2, 3, 4, 5]
}

classifier_info = {
                   "nbatches" : 50,
                   "nepochs" : 200,
                   "nfeatures" : 2000,
                   "ngrams" : 2,
                   "alpha" : 0.001,
                   "embedding_dim" : 2000,
                   "hidden_dim" : 200,
                   "output_dim" : 5
}

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])

extractor = data_handler.generate_bag_of_ngrams_extractor(train_documents, classifier_info["nfeatures"], classifier_info["ngrams"])
pickle.dump(extractor, open(PATH_TO_EXTRACTOR, "wb"))

train_input = data_handler.generate_input(train_documents, extractor)
val_input = data_handler.generate_input(val_documents, extractor)

train_label_input = np.array(train_labels)
val_label_input = np.array(val_labels)

train_label_class_indices = data_handler.labels_to_indices(train_label_input, data_info["class_labels"])

print("#################################################################### \n")
print("TRAINING: PYTORCH FEED FORWARD\n")
print("#################################################################### \n")

pytorch_ff_classifier = feed_forward_pt.FeedForwardClassifier(data_info, classifier_info)
pytorch_ff_classifier.train(train_input, train_label_class_indices)
pickle.dump(pytorch_ff_classifier, open(PATH_TO_CLASSIFIER, "wb"))

print("#################################################################### \n")
print("TESTING: PYTORCH FEED FORWARD\n")
print("#################################################################### \n")

predictions, actual = pytorch_ff_classifier.test(val_input, val_label_input)
print("predictions[0:100]", predictions[0:100])
print("actual[0:100]", actual[0:100])
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
