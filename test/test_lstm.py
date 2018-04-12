import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import test_utils
import pre_process
import glove_extractor
import lstm
import pickle
import json
import numpy as np

print("#################################################################### \n")
print("LOADING FROM FILE: LSTM\n")
print("####################################################################\n")

train_documents = []
train_labels = []

with open("review.json", 'r', encoding='utf8') as file:
    for i, line in enumerate(file):
        if i == 100000:
            break
        data = json.loads(line)
        train_documents.append(data["text"])
        train_labels.append(data["stars"])

test_documents = train_documents[-10:]
test_labels = train_labels[-10:]
train_documents = train_documents[0:90]
train_labels = train_labels[0:90]

print("#################################################################### \n")
print("TRAINING: LSTM\n")
print("#################################################################### \n")

print("Pre_processing...")

cleaner = pre_process.DocumentCleaner()

train_documents = cleaner.clean(train_documents[0:int(len(train_documents) // 1)])
test_documents = cleaner.clean(test_documents[0:int(len(test_documents) // 1)])

train_labels = train_labels[0:int(len(train_labels) // 1)]
test_labels = test_labels[0:int(len(test_labels) // 1)]

print("Extracting features...")

NFEATURES = 200
EMBED_SIZE = 300
HIDDEN_DIM = 200
class_list = [1, 2, 3, 4, 5]
NBATCHES = 10
NEPOCHS = 10
ALPHA = 0.01
NLABELS = len(class_list)

extractor = glove_extractor.GloveFeatureExtractor()
token2id = extractor.create_token2id(train_documents, NFEATURES)
print("token2id", token2id)
embeddings = extractor.extract_glove_embeddings('./glove.42B.300d.txt', NFEATURES, EMBED_SIZE, token2id)
print("embeddings", embeddings)

pickle.dump(extractor, open("../data/lstm_extractor.p", "wb"))

train_input = extractor.extract_ids(train_documents)

print("train_input: ", train_input)

print("Training")

lstm_classifier = lstm.LSTMClassifier(EMBED_SIZE, HIDDEN_DIM, NFEATURES, NLABELS)
lstm_classifier.set_embedding_weights(embeddings)
lstm.train(lstm_classifier, train_input, train_labels, ALPHA, NEPOCHS, NBATCHES)
