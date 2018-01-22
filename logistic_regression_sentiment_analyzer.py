import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class LogisticRegressionSAClassifier:

    LAMBDA = 1
    ALPHA = 2
    NGRAMS = 2
    NITERATIONS = 2000
    NFEATURES = 2000

    def __init__(self, training_set, training_labels):
        self.training_set = training_set
        self.training_labels = training_labels
        self.features = {}
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
        return loss - reg

    def grads(self, x, w, y, h):
        m = x.shape[1]
        grads = 1 / float(m) * np.dot(x, (h - y))

        reg = self.LAMBDA / float(m) * w
        reg[0] = 0

        return grads + reg

    def compute_ngrams(self, document, n):
        tokens = word_tokenize(document)
        if n == 1:
            return tokens
        else:
            return ngrams(tokens, n)

    def build_feature_set(self, data):
        print("starting build_feature_set")
        for n in range(self.NGRAMS):
            ngrams = []

            for i in range(len(data)):
                document = data[i]
                ngrams.extend(self.compute_ngrams(document, n + 1))

            fdist = FreqDist(ngram for ngram in ngrams)

            if self.NFEATURES != None:
                self.features[n + 1] = [i[0] for i in fdist.most_common(self.NFEATURES // self.NGRAMS)]
            else:
                self.features[n + 1] = [i[0] for i in fdist]

        print("finished build_feature_set")
        return self.features

    def doc_features(self, document):
        all_grams_features = []

        for n in range(self.NGRAMS):
            features = []
            ngrams = set(self.compute_ngrams(document, n + 1))

            for ngram in self.features[n + 1]:
                features.append(1 if ngram in ngrams else 0)

            all_grams_features.extend(features)

        return all_grams_features

    def input_matrix(self, documents):
        print("building input matrix")
        feature_set = []

        for document in documents:
            feature_set.append(self.doc_features(document))

        ones = np.ones((1, len(documents)))
        print("feature_set shape", np.array(feature_set).T.shape)
        return np.concatenate((ones, np.array(feature_set).T), axis=0)

    def batch_gradient_descent(self, data):
        for i in range(self.NITERATIONS):
            h = self.predict(self.w, data)
            grads = self.grads(data, self.w, self.training_labels, h)
            self.w = self.w - self.ALPHA * grads

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", self.loss(data, self.w, self.training_labels, h))

        print("after training, weights are: ")
        print(self.w)

    def stochastic_gradient_descent(self, data):
        print("starting stochastic")
        m = data.shape[1]

        for i in range(self.NITERATIONS):
            for j in range(m):
                grads = (self.predict(self.w, data[:,j]) - self.training_labels[j]) * data[:,j]
                reg = self.LAMBDA * self.w / float(m)
                reg[0] = 0
                self.w = self.w - self.ALPHA * (grads + reg)

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", self.loss(data, self.w, self.training_labels, self.predict(self.w, data)))

        print("after training, weights are: ")
        print(self.w)

    def train(self, method):
        data_input = self.input_matrix(self.training_set)
        self.w = np.zeros(self.NFEATURES + 1)

        if method == "batch":
            self.batch_gradient_descent(data_input)
        else:
            self.stochastic_gradient_descent(data_input)

    def test(self, data, labels):
        data_input = self.input_matrix(data)
        c = self.classify(self.w, data_input)
        n_examples = len(labels)
        n_correct = 0

        for i in range(n_examples):
            if c[i] == labels[i]:
                n_correct = n_correct + 1

        return n_correct / float(n_examples)


terminators = {',', '.', '!', ';', ':', '?', '\n'}
negations = {"not", "no", "never", "n't"}
NEEDED_STOPWORDS = ['over','only','very','not','no']
STOPWORDS = set(stopwords.words('english')) - set(NEEDED_STOPWORDS)
NEG_CONTRACTIONS = [
    (r'aren\'t', 'are not'),
    (r'can\'t', 'can not'),
    (r'couldn\'t', 'could not'),
    (r'daren\'t', 'dare not'),
    (r'didn\'t', 'did not'),
    (r'doesn\'t', 'does not'),
    (r'don\'t', 'do not'),
    (r'isn\'t', 'is not'),
    (r'hasn\'t', 'has not'),
    (r'haven\'t', 'have not'),
    (r'hadn\'t', 'had not'),
    (r'mayn\'t', 'may not'),
    (r'mightn\'t', 'might not'),
    (r'mustn\'t', 'must not'),
    (r'needn\'t', 'need not'),
    (r'oughtn\'t', 'ought not'),
    (r'shan\'t', 'shall not'),
    (r'shouldn\'t', 'should not'),
    (r'wasn\'t', 'was not'),
    (r'weren\'t', 'were not'),
    (r'won\'t', 'will not'),
    (r'wouldn\'t', 'would not'),
    (r'ain\'t', 'am not')
]


def add_negations(word_list):
    in_negation_zone = False
    for i in range(len(word_list)):
        word = word_list[i]
        if in_negation_zone:
            word_list[i] = "NOT_" + word
        if word in negations or word[-3:] in negations:
            in_negation_zone = not in_negation_zone
        if word[-1] in terminators:
            in_negation_zone = False

    return word_list

def remove_terminators(word_list):
    for i in range(len(word_list)):
        word = word_list[i]
        last = len(word) - 1

        while last >= 0 and (word[last] in terminators or word[last] == '\n'):
            last = last - 1

        word_list[i] = word[0:last + 1]

    return word_list

def remove_null_words(word_list):
    length = len(word_list)
    i = 0

    while i < length:
        if word_list[i] == "":
            del word_list[i]
            length = length - 1
        else:
            i = i + 1

    return word_list

def remove_stop_words(word_list):
    length = len(word_list)
    i = 0

    while i < length:
        if word_list[i] in STOPWORDS:
            del word_list[i]
            length = length - 1
        else:
            i = i + 1

    return word_list


def pre_process(documents):
    print("in pre_process")
    for i in range(len(documents)):
        document = documents[i].lower()

        for word in NEG_CONTRACTIONS:
            document = re.sub(word[0], word[1], document)

        word_list = document.split(' ')
        remove_null_words(word_list)
        add_negations(word_list)
        remove_terminators(word_list)
        #remove_stop_words(word_list)
        documents[i] = " ".join(word_list)

    print("done with pre_process")
    return documents

def strip_labels(documents):
    texts = []
    labels = []
    for d in documents:
        if d.split(' ', 1)[0] == "__label__1":
            labels.append(0)
        else:
            labels.append(1)

        texts.append(d.split(' ', 1)[1])

    return texts, np.array(labels).T

with open("test.ft.txt", 'r', encoding='utf8') as file:
    test_reviews = tuple(file)

with open("train.ft.txt", 'r', encoding='utf8') as file:
    train_reviews = tuple(file)


    #m = x.shape[1]
    #grads = 1 / float(m) * np.dot(x, (h - y))

    #reg = self.LAMBDA / float(m) * w
    #reg[0] = 0

h = np.array([0.5, 0.5, 0.5, 0.5])
y = np.array([0.75, 0.25, 0.1, 0.1])
x = np.array([[3, 4, 5, 6], [1, 2, 1, 2]])
w = np.array([3, 4])


train_texts, train_labels = strip_labels(train_reviews[0:int(len(train_reviews) / 100)])
test_texts, test_labels = strip_labels(test_reviews[0:int(len(test_reviews) / 100)])

ALPHA_VALUES = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 3, 5, 10, 50, 100]

#ALPHA of ~1.5 / 2 is best 

lr_classifier = LogisticRegressionSAClassifier(pre_process(train_texts), train_labels)

for val in ALPHA_VALUES:
    lr_classifier.ALPHA = val
    lr_classifier.train("batch")
    print("ALPHA: ")
    print(val)
    print("Train Set Accuracy:")
    print(lr_classifier.test(train_texts, train_labels))
    print("Test Set Accuracy:")
    print(lr_classifier.test(test_texts, test_labels))
