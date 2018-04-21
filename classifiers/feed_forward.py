import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import utils

class FeedForwardClassifier:

    def __init__(self, data_info, classifier_info):
        self.data_info = data_info
        self.classifier_info = classifier_info
        self.class_labels = data_info["class_labels"]
        self.niterations = classifier_info["niterations"]
        self.Lambda = classifier_info["lambda"]
        self.alpha = classifier_info["alpha"]
        self.layer_dims = classifier_info["layers_dims"]
        self.batch_size = classifier_info["batch_size"]
        self.method = classifier_info["method"]

    def train(self, X, Y):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
        every document in the training set
        method : a string indicating the training method to be used

        Returns:

        parameters : a dictionary containing the trained parameters [ Wl and bl ] for each layer of the network
        """
        one_hot_Y = self.convert_to_one_hot(Y)
        self.parameters = self.initialize_params(self.layer_dims)

        if self.method == "batch":
            self.parameters = self.batch_gradient_descent(X, one_hot_Y, self.parameters, self.alpha, self.niterations, self.Lambda, self.batch_size)
        elif self.method == "stochastic":
            self.parameters = self.stochastic_gradient_descent(X, one_hot_Y, self.parameters, self.alpha, self.niterations, self.Lambda)

        return self.parameters

    def convert_to_one_hot(self, Y):
        """
        Arguments:
        Y : a numpy array of dimension [number of documents] x 1, containing the class labels of
            a set of documents

        Stores:

        one_hot.w : a numpy array of dimension [number of class labels] x [number of documents], containing the "one-hot"
                    representation of all of the class labels for every document in the document set
        """
        one_hot = np.zeros((self.layer_dims[-1], len(Y)))

        for i in range(len(Y)):
            one_hot[Y[i] - 1, i] = 1


        return one_hot

    def batch_gradient_descent(self, X, Y, parameters, learning_rate, num_iterations, Lambda, batch_size):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y : a numpy array of dimension [number of classes] x [number of training examples], containing the
            "one-hot" class labels of every document in the training set
        parameters : an empty dictionary, ultimately to contain the parameters [Wl] and [bl] for each layer
                     of the network
        learning_rate : the size of each update step
        num_iterations : the number of iterations

        Stores:
        parameters : a dictionary containing the parameters [Wl] and [bl] for each layer of the network
        """
        losses = []

        m = Y.shape[1]
        num_batches = X.shape[1] // batch_size

        for i in range(0, num_iterations):
            for n in range(0, num_batches):
                X_batched = X[:, n * batch_size : (n + 1) * batch_size]
                Y_batched = Y[:, n * batch_size : (n + 1) * batch_size]
                AL, caches = self.forward_prop(X_batched, parameters)
                grads = self.backward_prop(AL, Y_batched, caches)
                parameters = self.update_parameters(parameters, grads, learning_rate, m, Lambda)

            if i % 10 == 0:
                AL, caches = self.forward_prop(X, parameters)
                loss = self.loss(AL, Y, parameters)
                print ("Loss after iteration %i: %f" %(i, loss))
                losses.append(loss)

        return parameters

    def stochastic_gradient_descent(self, X, Y, parameters, learning_rate, num_iterations, Lambda):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
            every document in the training set
        parameters : an empty dictionary, ultimately to contain the parameters [Wl] and [bl] for each layer
                     of the network
        learning_rate : the size of each update step
        num_iterations : the number of iterations

        Stores:
        parameters : a dictionary containing the parameters [Wl] and [bl] for each layer of the network
        """
        losses = []
        m = X.shape[1]

        for i in range(0, num_iterations):

            for j in range(0, m):
                data_in = np.reshape(X[:,j], (X[:,j].shape[0], 1))
                label = np.array([Y[j]])
                AL, caches = self.forward_prop(data_in, parameters)
                grads = self.backward_prop(AL, label, caches)
                parameters = self.update_parameters(parameters, grads, learning_rate, m, Lambda)

            if i % 100 == 0:
                loss = self.loss(AL, Y, parameters)
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
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
        caches.append(cache)
        assert(AL.shape == (5,X.shape[1]))

        return AL, caches

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
            A, activation_cache = utils.lrelu(Z)
        else:
            A, activation_cache = utils.nn_softmax(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

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

    def backward_prop(self, AL, Y, caches):
        """
        Arguments:
        AL - a numpy array containing the outputs generated by the forward pass through the network; dimension (number of examples, 1)
        Y - one-hot vectors labeling of the training data; dimension (number of examples, number of classes)
        caches - a python list of the caches stored in the forward pass of the algorithm; length (number of layers)

        Returns:
        grads - a python dictionary storing the gradients for the weights, biases, and outputs of each layers
        """

        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        current_cache = caches[L-1]
        linear_cache, activation_cache = current_cache
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(AL - Y, linear_cache)

        for l in reversed(range(L - 1)):
            dA = grads["dA" + str(l + 2)]
            current_cache = caches[l]
            grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.linear_activation_backward(dA, current_cache, "relu")

        return grads

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
            dZ = utils.lrelu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "softmax":
            dZ = utils.softmax_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

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

        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, keepdims = True, axis = 1)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def loss(self, AL, Y, parameters):
        """
        Arguments:
        AL - a numpy array containing the predictions of the network; dimension (number of examples, 1)
        Y - the actual labels of the training set; dimension (number of exmples, 1)

        Returns:
        cost - the cross entropy log loss, a scalar
        """
        m = Y.shape[1]

        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        loss = -1 * np.sum(logprobs) / m

        reg = 0
        L = len(parameters) // 2

        for l in range(1, L + 1):
            w = parameters["W" + str(l)]
            reg += self.Lambda / float(2 * m) * np.sum(np.multiply(w, w))

        return loss + reg

    def update_parameters(self, params, grads, learning_rate, m, Lambda):
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
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * (grads["dW" + str(l)] + Lambda / float(m) * params["W" + str(l)])
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]

        return params

    def test(self, X, Y):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of test examples], containing the
            values of every feature in every document in the test set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y -  a numpy array containing the actual class labels for each document in the test set

        Returns:
        predictions : a numpy array containing the predicted class labels for each document
                      in the test set (mapped by self.classify())
        Y : a numpy array containing the actual class labels for each document in the test set
        """
        predictions = np.array([])

        for i in range(X.shape[1]):
            doc_data = X[:,i].reshape(X[:,i].shape[0], 1)
            predictions = np.append(predictions,self.classify(doc_data))

        predictions = predictions.reshape(Y.shape)

        return predictions, Y

    def classify(self, data):
        """
        Arguments:
        data     : a numpy array of dimension [number of features] x [1], containing the
                   values of every feature in a single document; includes a row of 1s which pair
                   with the w[0] (the bias)

        Returns:
        max : a scalar containing the predicted class label for the given document
        """
        h = self.predict(data)
        max = np.argmax(h)
        return self.class_labels[max]

    def predict(self, data):
        """
        Arguments:
        data     : a numpy array of dimension [number of features] x [1], containing the
                   values of every feature in a single document

        Returns:
        h : the output of the classifier when applied to the given set of data
        """
        h = self.forward_prop(data, self.parameters)[0]
        return h
