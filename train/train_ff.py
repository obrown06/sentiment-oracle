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

# (unbalanced) n_samples_train: 146000
# (unbalanced) n_samples_test: 10000

data_info = {"source" : "ROTTEN_TOMATOES",
             "path" : "../data/train.tsv",
             "is_balanced" : True,
             "n_samples_train" : 6000,
             "n_samples_val" : 500,
             "n_samples_test" : 500,
             "class_labels" : [1, 2, 3, 4, 5]
}

classifier_info = {
                   "method" : "batch",
                   "batch_size" : 100,
                   "nfeatures" : 2000,
                   "ngrams" : 2,
                   "niterations" : 500,
                   "alpha" : 0.001,
                   "lambda" : 0.5,
                   "layers_dims" : [2000, 200, 5]

}

train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index = data_handler.load_data(data_info["source"], data_info["path"], data_info["n_samples_train"], data_info["n_samples_val"], data_info["n_samples_test"], data_info["class_labels"], is_balanced=data_info["is_balanced"])

extractor = data_handler.generate_bow_extractor(train_documents, classifier_info["nfeatures"], classifier_info["ngrams"])
pickle.dump(extractor, open("../pickle/ff_extractor.p", "wb"))

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
pickle.dump(ff_classifier, open("../pickle/ff_classifier.p", "wb"))

print("#################################################################### \n")
print("VALIDATING: FEED FORWARD\n")
print("#################################################################### \n")

predictions, actual = ff_classifier.test(val_input, val_label_input)
accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
