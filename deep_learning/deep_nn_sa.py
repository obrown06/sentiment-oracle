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

class DeepNet:

    def train(X, Y, learning_rate, num_iterations, type):
        self.parameters = initialize_parameters(layer_dims)

        if type == "batch":
            self.parameters = batch_gradient_descent(X, Y, self.parameters, learning_rate, num_iterations)
        elif type == "stochastic":
            self.parameters = stochastic_gradient_descent(X, Y, self.parameters, learning_rate, num_iterations)

        return self.parameters

    def classify(outputs, threshold):

        return outputs[outputs >= threshold]

    def test(X, Y, threshold):
        AL, caches = forward_prop(X, self.parameters)

        predictions = classify(AL, threshold)

        tp = predictions[predictions == 1 and Y == 1].sum()
        fp = predictions[predictions == 1 and Y == 0].sum()
        tn = predictions[predictions == 0 and Y == 0].sum()
        fn = predictions[predictions == 0 and Y == 1].sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fn + fp)

        return accuracy, precision, recall, specificity

    def batch_gradient_descent(X, Y, parameters, learning_rate, num_iterations):
        losses = []

        for i in range(0, num_iterations):

            AL, caches = forward_prop(X, parameters)
            loss = loss(AL, Y)
            grads = backward_prop(AL, Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)

            if i % 100 == 0:
                print ("Loss after iteration %i: %f" %(i, loss))
                losses.append(loss)

        return parameters

    def stochastic_gradient_descent(X, Y, parameters, learning_rate, num_iterations):
        losses = []
        m = X.shape[1]

        for i in range(0, num_iterations):

            for j in the range(0, m):
                AL, caches = forward_prop(X[:,j], parameters)
                loss = loss(AL, Y)
                grads = backward_prop(AL, Y, caches)
                parameters = update_parameters(parameters, grads, learning_rate)

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

    def linear_activation_forward(A_prev, W, b, activation):
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

        Z, linear_cache = linear_forward(A_prev, W, b)

        if activation == "relu":
            A, activation_cache = relu(Z)
        else:
            A, activation_cache = sigmoid(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(X, parameters):
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
            A_prev = A
            A, cache = linear_activation_forward(A_prev, parameters['W' + l], parameters['b' + l], "relu")
            caches.append(cache)

        AL, cache = linear_activation_forward(A, parameters['W' + l], parameters['b' + l], "sigmoid")

        caches.append(cache)

        return AL, caches

    def loss(AL, Y):
        """
        Arguments:
        AL - a numpy array containing the predictions of the network; dimension (number of examples, 1)
        Y - the actual labels of the training set; dimension (number of exmples, 1)

        Returns:
        cost - the cross entropy log loss, a scalar
        """

        m = Y.shape[1]

        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        loss = -1 / m * np.sum(logprobs)

        cost = np.squeeze(loss)

        return cost

    def linear_backward(dZ, cache):
        """
        Arguments:
        dZ - a numpy array containing the derivatives of the linear outputs of the network; dimension (size of layer, number of examples)
        cache - a python dictionary containing the previous layer's outputs and this layer's weights and biases

        Returns:
        dA_prev - a numpy array containing the derivatives of this activated outputs of the previous layer; dimension (size of previous layer, number of examples)
        dW - a numpy array containing the derivatives of the weights of this layer; dimension (size of current layer, size of previous layer)
        db - a numpy array containing the derivatives of the biases of this layer; dimension (size of current layer, 1)
        """

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        dB = 1 / m * np.sum(dZ, keepdims = True, axis = 1)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(dA, cache, activation):
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
            dZ = relu(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def backward_prop(AL, Y, caches):
        """
        Arguments:
        AL - a numpy array containing the outputs generated by the forward pass through the network; dimension (number of examples, 1)
        Y - the labels of the training data; dimension (number of examples, 1)
        caches - a python list of the caches stored in the forward pass of the algorithm; length (number of layers)

        Returns:
        grads - a python dictionary storing the gradients for the weights, biases, and outputs of each layers
        """

        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

        for l in reversed(range(L - 1)):
            dA = grads["dA" + str(l + 2)]
            current_cache = caches[l]
            grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(dA, current_cache, "relu")

        return grads


    def update_parameters(params, grads, learning_rate):
        """
        Arguments:
        params - a python dictionary containing the weights and biases of the layers in the network
        grads - a python dictionary containing the gradients of the weights and biases of the layers in the network.
        learning_rate - a scalar indicating the learning_rate hyperparameter of the network

        Returns:
        params - the updated python dictionary of weights and biases
        """

        L = len(parameters) // 2

        for l in range(1, L + 1):
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["W" + str(l)]
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["b" + str(l)]

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
