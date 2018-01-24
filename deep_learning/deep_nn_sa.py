# Tasks:

# initialize_parameters
# linear_forward
# linear_activation_forward
# forward_prop
# loss
# linear_backward
# linear_activation_backward
# backward_prop
# update_parameters

import re
import numpy as np

class DeepNeuralNet:

    # Intakes a list of layer dimensions

    def __init__(self, layer_dims):

        self.training_set = training_set
        self.training_labels = training_labels
        self.features = {}
        self.build_feature_set(training_set)

    def initialize_params(self, layer_dims):
        self.parameters = {}

        for l in range(1, len(layer_dims)):
            self.parameters['W' + l] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            self.parameters['b' + l] = np.zeros((layer_dims[l], 1))

    def linear_forward(self, A, w, b):

    def linear_activation_forward(A, w, b):

    def forward_prop(X, parameters):

    def loss(h, Y):

    def linear_backward(A, w):

    def linear_activation_backward(A, w, b):

    def backward_prop(X, parameters):

    def update_parameters(params, grads, learning_rate):





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
    lr_classifier.LAMBDA = val

    lr_classifier.train("batch")
    print("LAMBDA: ")
    print(val)
    print("Train Set Accuracy:")
    print(lr_classifier.test(train_texts, train_labels))
    print("Test Set Accuracy:")
    print(lr_classifier.test(test_texts, test_labels))
