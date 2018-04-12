import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pre_process
import deep_net
import bow_extractor
import pickle
import json
import numpy as np
import pytorch_feed_forward

print("#################################################################### \n")
print("LOADING FROM FILE: PYTORCH FEED FORWARD\n")
print("####################################################################\n")

train_documents = []
train_labels = []

with open("review.json", 'r', encoding='utf8') as file:
    for i, line in enumerate(file):
        if i == 1000000:
            break
        data = json.loads(line)
        train_documents.append(data["text"])
        train_labels.append(data["stars"])

test_documents = train_documents[-100000:]
test_labels = train_labels[-100000:]
train_documents = train_documents[0:900000]
train_labels = train_labels[0:900000]

print("#################################################################### \n")
print("TRAINING: PYTORCH FEED FORWARD\n")
print("#################################################################### \n")

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

pickle.dump(extractor, open("../data/pytorch_ff_extractor.p", "wb"))

train_input = extractor.extract_features(train_documents, feature_set)
test_input = extractor.extract_features(test_documents, feature_set)
class_list = [1, 2, 3, 4, 5]
train_label_input = np.array(train_labels)
test_label_input = np.array(test_labels)

train_label_input = train_label_input - 1
test_label_input = test_label_input - 1

print("Training...")

# Alpha = 0.001 and NEPOCHS = 200 and NBATCHES = 50 and optim = SGD gives 0.675 accuracy, 0.8515 polarity, 0.93 near accuracy

NEPOCHS = 200
ALPHA = 0.001
NBATCHES = 50
input_dim = NFEATURES
hidden_dim = 200
output_dim = 5
class_list = [1, 2, 3, 4, 5]

pytorch_net = pytorch_feed_forward.FeedForwardClassifier(input_dim, hidden_dim, output_dim)
pytorch_feed_forward.train(pytorch_net, train_input, train_label_input, ALPHA, NEPOCHS, NBATCHES)

pickle.dump(pytorch_net, open("../classifiers/pytorch_ff_classifier.p", "wb"))

print("Testing...\n")

with open('../classifiers/pytorch_ff_classifier.p', 'rb') as pickle_file:
    pytorch_classifier = pickle.load(pickle_file)

predictions, actual = pytorch_feed_forward.test(pytorch_net, class_list, test_input, test_label_input + 1)
print("predictions: ", predictions)
print("actual", actual)

accuracy, near_accuracy, accurate_polarity = test_utils.multiclass_accuracy(predictions, actual)

print("####################################################################\n")

print("RESULTS:\n")
print("Accuracy: ", accuracy)
print("Near Accuracy: ", near_accuracy)
print("Accurate Polarity: ", accurate_polarity)

print("####################################################################")
