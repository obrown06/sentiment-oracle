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
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class DeepNet:

    def train(self, X, Y, layer_dims, learning_rate, num_iterations, type):
        print("X")
        print(X)
        print("Y")
        print(Y)
        print("Layer dims")
        print(layer_dims)
        self.parameters = self.initialize_params(layer_dims)

        if type == "batch":
            self.parameters = self.batch_gradient_descent(X, Y, self.parameters, learning_rate, num_iterations)
        elif type == "stochastic":
            self.parameters = self.stochastic_gradient_descent(X, Y, self.parameters, learning_rate, num_iterations)

        return self.parameters

    def classify(self, outputs, threshold):
        print("inputs", outputs)
        outputs[outputs >= threshold] = 1
        outputs[outputs < threshold] = 0

        print("outputs", outputs)

        return outputs

    def test(self, X, Y, threshold):
        AL, caches = self.forward_prop(X, self.parameters)

        predictions = self.classify(AL, threshold)
        print("Y", Y)
        print("predictions", predictions)

        tp = ((predictions == 1) & (predictions == Y)).sum()
        fp = ((predictions == 1) & (predictions != Y)).sum()
        tn = ((predictions == 0) & (predictions == Y)).sum()
        fn = ((predictions == 0) & (predictions != Y)).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fn + fp)

        return accuracy, precision, recall, specificity

    def batch_gradient_descent(self, X, Y, parameters, learning_rate, num_iterations):
        losses = []

        for i in range(0, num_iterations):
            AL, caches = self.forward_prop(X, parameters)
            #print("AL")
            #print(AL)
            grads = self.backward_prop(AL, Y, caches)
            #print("grads")
            #print(grads)
            parameters = self.update_parameters(parameters, grads, learning_rate)

            if i % 1000 == 0:
                loss = self.loss(AL, Y)
                print ("Loss after iteration %i: %f" %(i, loss))
                losses.append(loss)
                print("parameters", parameters)

        print("parameters at end", parameters)
        return parameters

    def stochastic_gradient_descent(self, X, Y, parameters, learning_rate, num_iterations):
        losses = []
        m = X.shape[1]

        for i in range(0, num_iterations):

            for j in range(0, m):
                AL, caches = self.forward_prop(X[:,j], parameters)
                loss = self.loss(AL, Y)
                grads = self.backward_prop(AL, Y, caches)
                parameters = self.update_parameters(parameters, grads, learning_rate)

                if i % 100 == 0:
                    print ("Loss after iteration %i: %f" %(i, loss))
                    losses.append(loss)

        return parameters

    def initialize_params(self, layer_dims):
        """
        Arguments:
        layer_dims - a python array containing integers specifying the dimension of each layer of the network.

        Returns (and stores as a field):
        A dictionary containing the parameters at each layer of the network:
                Wl - a numpy array of dimension (layer_dims[l - 1], layer_dims[l]) containing the layer's weights
                bl - a numpy array of dimension (layer_dims[l]) containing the layer's biases
        """

        parameters = {}

        for l in range(1, len(layer_dims)):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        print("starting params")
        print(parameters)
        return parameters

    def linear_forward(self, A, W, b):
        """
        Arguments:
        A - a numpy array containing the previous layer's outputs; dimension (size of previous layer, number of examples)
        W - a numpy array containing the weights of each node in this layer; dimension (size of current layer, size of previous layer)
        b - a numpy array containing the biases of each node in this layer; dimension (size of current layer, 1)

        Returns:
        Z - a numpy array containing the linear output of each node; dimension (size of current layer, number of examples)
        cache - a python dictionary containing A, W, b; stored for later use in backprop
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Arguments:
        A_prev - a numpy array containing the previous layer's outputs; dimension (size of previous layer, number of examples)
        W - a numpy array containing the weights of each node in this layer; dimension (size of current layer, size of previous layer)
        b - a numpy array containing the biases of each node in this layer; dimension (size of current layer, 1)
        activation - this layer's activation function, stored as a lambda expression

        Returns:
        Z - a numpy array containing the output of each node after activation; dimension (size of current layer, number of examples)
        cache - a python dictionary containing the linear cache and the activation cache.
        """

        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "relu":
            A, activation_cache = self.relu(Z)
        else:
            #print("forward pass before sigmoid, Z is: ", Z)
            A, activation_cache = self.sigmoid(Z)
            #print("forward_pass after sigmoid, A is: ", A)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self, X, parameters):
        """
        Arguments:
        X - a numpy array containing the inputs to the network; dimension (size of input layer, number of examples)
        parameters - a dictionary containing the weights and biases of the network, stored as numpy arrays.

        Returns:
        A - a numpy array containing the outputs of the last layer of the network
        caches - a python dictionary containing the cached inputs, weights, and biases of every layer in the network.
        """

        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            #print("A for layer: ", l)
            #print(A)
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")

        caches.append(cache)
        assert(AL.shape == (1,X.shape[1]))

        return AL, caches

    def loss(self, AL, Y):
        """
        Arguments:
        AL - a numpy array containing the predictions of the network; dimension (number of examples, 1)
        Y - the actual labels of the training set; dimension (number of exmples, 1)

        Returns:
        cost - the cross entropy log loss, a scalar
        """

        m = Y.shape[0]

        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        loss = -1 * np.sum(logprobs) / m

        cost = np.squeeze(loss)

        return cost

    def linear_backward(self, dZ, cache):
        """
        Arguments:
        dZ - a numpy array containing the derivatives of the linear outputs of the network; dimension (size of layer, number of examples)
        cache - a python dictionary containing the previous layer's outputs and this layer's weights and biases

        Returns:
        dA_prev - a numpy array containing the derivatives of this activated outputs of the previous layer; dimension (size of previous layer, number of examples)
        dW - a numpy array containing the derivatives of the weights of this layer; dimension (size of current layer, size of previous layer)
        db - a numpy array containing the derivatives of the biases of this layer; dimension (size of current layer, 1)
        """
        #print("in linear_backward, dZ: ", dZ)
        A_prev, W, b = cache
        #print("in linear_backward, A_prev", A_prev)
        m = A_prev.shape[1]
        #print("in linear_backward, dW before division")
        #print(np.dot(dZ, A_prev.T))

        dW = 1 / m * np.dot(dZ, A_prev.T)

        #print("in linear_backward, dW after division", dW)
        db = 1 / m * np.sum(dZ, keepdims = True, axis = 1)
        #print("db before division")
        #print(np.sum(dZ, keepdims = True, axis = 1))
        #print("db after division")
        #print(db)
        dA_prev = np.dot(W.T, dZ)

        #print("in linear_backward, dA_prev", dA_prev)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Arguments:
        dA - a numpy array containing the derivatives of this layer's post-activation outputs; dimension (size of current layer, number of examples)
        cache - a python dictionary containing the linear and activation caches of this layer
        activation - a string identifying this layer's activation function

        Returns:
        dA_prev - a numpy array containing the derivatives of this activated outputs of the previous layer; dimension (size of previous layer, number of examples)
        dW - a numpy array containing the derivatives of the weights of this layer; dimension (size of current layer, size of previous layer)
        db - a numpy array containing the derivatives of the biases of this layer; dimension (size of current layer, 1)
        """

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            #print("before backward sigmoid, dA", dA)
            #print("before backward sigmoid, Z", activation_cache)
            dZ = self.sigmoid_backward(dA, activation_cache)
            #print("after backward sigmoid, dZ: ", dZ)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            #print("after linear_backward, dA_prev is: ", dA_prev)

        return dA_prev, dW, db

    def lrelu(self, Z):
        A = np.where(Z > 0, Z, 0.1 * Z)
        return A, Z

    def relu(self, Z):
        A = np.where(Z > 0, Z, 0)
        return A, Z

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A, Z

    def sigmoid_backward(self, dA, A):
        return np.multiply(dA, np.multiply(A, 1 - A))

    def lrelu_backward(self, dA, Z):
        return np.multiply(dA, np.where(Z > 0, 1, 0.1))

    def relu_backward(self, dA, Z):
        return np.multiply(dA, np.where(Z > 0, 1, 0))

    def backward_prop(self, AL, Y, caches):
        """
        Arguments:
        AL - a numpy array containing the outputs generated by the forward pass through the network; dimension (number of examples, 1)
        Y - the labels of the training data; dimension (number of examples, 1)
        caches - a python list of the caches stored in the forward pass of the algorithm; length (number of layers)

        Returns:
        grads - a python dictionary storing the gradients for the weights, biases, and outputs of each layers
        """

        #print("AL")
        #print(AL)
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        #print("Y", Y)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        #print("dAL", dAL)
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")

        for l in reversed(range(L - 1)):
            #print("layer ", l)
            dA = grads["dA" + str(l + 2)]
            #print("dA", dA)
            current_cache = caches[l]
            grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.linear_activation_backward(dA, current_cache, "relu")
            #print("dW", grads["dW" + str(l + 1)])
            #print("db", grads["db" + str(l + 1)])
        return grads


    def update_parameters(self, params, grads, learning_rate):
        """
        Arguments:
        params - a python dictionary containing the weights and biases of the layers in the network
        grads - a python dictionary containing the gradients of the weights and biases of the layers in the network.
        learning_rate - a scalar indicating the learning_rate hyperparameter of the network

        Returns:
        params - the updated python dictionary of weights and biases
        """

        L = len(params) // 2

        for l in range(1, L + 1):
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]

        return params

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

def compute_ngrams(document, n):
    tokens = word_tokenize(document)
    if n == 1:
        return tokens
    else:
        return ngrams(tokens, n)

def build_feature_set(data):
    features = {}
    NFEATURES = 200
    NGRAMS = 2
    print("starting build_feature_set")
    for n in range(NGRAMS):
        ngrams = []

        for i in range(len(data)):
            document = data[i]
            ngrams.extend(compute_ngrams(document, n + 1))

        fdist = FreqDist(ngram for ngram in ngrams)

        features[n + 1] = [i[0] for i in fdist.most_common(NFEATURES // NGRAMS)]

    print("finished build_feature_set")
    return features

def doc_features(document, features):
    all_grams_features = []
    NGRAMS = 2

    for n in range(NGRAMS):
        doc_features = []
        ngrams = set(compute_ngrams(document, n + 1))

        for ngram in features[n + 1]:
            doc_features.append(1 if ngram in ngrams else 0)

        all_grams_features.extend(doc_features)

    return all_grams_features

def input_matrix(documents, features):
    print("building input matrix")
    feature_set = []

    for document in documents:
        feature_set.append(doc_features(document, features))

    print("input_matrix shape", np.array(feature_set).T.shape)
    return np.array(feature_set).T


    #m = x.shape[1]
    #grads = 1 / float(m) * np.dot(x, (h - y))

    #reg = self.LAMBDA / float(m) * w
    #reg[0] = 0

h = np.array([0.5, 0.5, 0.5, 0.5])
y = np.array([0.75, 0.25, 0.1, 0.1])
x = np.array([[3, 4, 5, 6], [1, 2, 1, 2]])
w = np.array([3, 4])


train_texts, train_labels = strip_labels(train_reviews[0:int(len(train_reviews) // 1000)])
test_texts, test_labels = strip_labels(test_reviews[0:int(len(test_reviews) / 1000)])

pre_processed_train_texts = pre_process(train_texts)
pre_processed_test_texts = pre_process(test_texts)

feature_set = build_feature_set(pre_process(train_texts))

input_matrix_train = input_matrix(pre_processed_train_texts, feature_set)
input_matrix_test = input_matrix(pre_processed_test_texts, feature_set)

dn = DeepNet()
layer_dims = [200, 7, 1]
dn.train(input_matrix_train, np.asarray(train_labels), layer_dims, 1, 50000, "batch")

accuracy, precision, recall, specificity = dn.test(input_matrix_test, np.asarray(test_labels), 0.5)

A = np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]])
W1 = np.array([[3, 4], [5, 6], [7, 8]]) 
b1 = np.array([1, 2, 3])[:, None]
W2 = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
b2 = np.array([-3, -2, -1])[:, None]
W3 = np.array([[3, 4, 5]])
b3 = np.array([1])[:, None]

parameters = dict()
parameters["W1"] = W1
parameters["b1"] = b1
parameters["W2"] = W2
parameters["b2"] = b2
parameters["W3"] = W3
parameters["b3"] = b3
#print(dn.forward_prop(A, parameters))
#print(dn.linear_activation_forward(A, W, b, "sigmoid"))
#print(dn.linear_forward(A, W, b))

print("accuracy", accuracy)
print("precision", precision)
print("recall", recall)
print("specificity", specificity)
