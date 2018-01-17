# Logistic Regression functions:

# pre_process()
# stochastic_descent()
# batch_descent()
# gradient_descent()
# loss()
# sigmoid()
# predict()
# test()
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class LogisticRegressionSAClassifier:

    LAMBDA = 0.1
    ALPHA = 0.1
    NITERATIONS = 1000
    NFEATURES = 2000

    def __init(self, training_set, training_labels):
        self.training_set = training_set
        self.training_labels = training_labels
        self.build_feature_set(training_set)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, w, x):
        return self.sigmoid(np.dot(x.T, w))

    def classify(self, w, x):
        h = self.predict(w, x)

        for i in range(len(h)):
            if h[i] >= 0.5:
                h[i] = 1
            else:
                h[i] = 0

        return h

    def loss(self, x, w, y, h):
        m = x.shape[1]
        loss = -1 / float(m) * (np.dot(np.log(h).T, y) + np.dot(np.log(1 - h).T, 1 - y))
        reg = self.LAMBDA / float(2 * m) * np.dot(w.T, w)
        return loss + reg

    def grads(self, x, w, y, h):
        m = x.shape[1]
        grads = 1 / float(m) * np.dot(x, (h - y))

        reg = self.LAMBDA / float(m) * w
        reg[0] = 0

        return grads + reg

    def build_feature_set(self, data):
        tokens = []

        for document in data:
            tokens.extend(word_tokenize(document))

        fdist = FreqDist(token for token in tokens)
        self.features = [i[0] for i in fdist.most_common(self.NFEATURES)]

        if len(self.features) < self.NFEATURES:
            self.NFEATURES = len(self.features)

        return self.features

    def doc_features(self, document):
        doc_features = np.zeros(self.NFEATURES)
        doc_tokens = word_tokenize(document)

        for i in range(self.NFEATURES):
            feature = self.features[i]
            if feature in doc_tokens:
                doc_features[i] = 1

        return doc_features

    def input_matrix(self, documents):
        feature_set = []

        for document in documents:
            feature_set.append(self.doc_features(document))

        ones = np.ones(self.NFEATURES)

        feature_set = np.append(ones, np.array(feature_set), axis=0)

        return feature_set.T

        #ones = np.ones(self.NFEATURES)

        #return np.append(np.array(feature_set), ones, axis=0)

    def train(self):
        data_input = self.input_matrix(self.training_set)
        self.w = np.zeros(self.NFEATURES)

        for i in range(self.NITERATIONS):
            h = predict(self.w, data_input)
            grads = self.grads(data_input, self.w, self.training_labels, h)
            w = w - self.ALPHA * grads

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", loss(data_input, self.w, self.training_labels, h))

    def test(self, data, labels):
        data_input = self.input_matrix(data)
        c = classify(self.w, data_input)
        n_examples = len(labels)
        n_correct = 0

        for i in range(n_examples):
            if c[i] == test_labels[i]:
                n_correct = n_correct + 1

        return n_correct / float(n_examples)

lrsa = LogisticRegressionSAClassifier()

docs = ["this is a document", "this is another document", "kaggle", "schmeh", "this is a meh", "keh", "a", "a"]
print(lrsa.build_feature_set(docs))

print(lrsa.input_matrix(docs))

x = np.array([[5, 2, 3], [-1, -2, -3]])
w = np.array([0.3, 0.4])

print(lrsa.classify(w, x))
