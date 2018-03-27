import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import utils

class DeepNetClassifier:

    def __init__(self, LAYER_DIMS, NITERATIONS = 2000, LAMBDA = 1, ALPHA = 0.2):
        self.NITERATIONS = NITERATIONS
        self.LAMBDA = LAMBDA
        self.ALPHA = ALPHA
        self.LAYER_DIMS = LAYER_DIMS

    def train(self, X, Y, method):
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
        self.parameters = self.initialize_params(self.LAYER_DIMS)

        if method == "batch":
            self.parameters = self.batch_gradient_descent(X, Y, self.parameters, self.ALPHA, self.NITERATIONS, self.LAMBDA)
        elif method == "stochastic":
            self.parameters = self.stochastic_gradient_descent(X, Y, self.parameters, self.ALPHA, self.NITERATIONS, self.LAMBDA)

        return self.parameters

    def batch_gradient_descent(self, X, Y, parameters, learning_rate, num_iterations, LAMBDA):
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

        for i in range(0, num_iterations):
            AL, caches = self.forward_prop(X, parameters)
            grads = self.backward_prop(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads, learning_rate, Y.shape[0], LAMBDA)

            if i % 100 == 0:
                loss = self.loss(AL, Y, parameters)
                print ("Loss after iteration %i: %f" %(i, loss))
                losses.append(loss)

        return parameters

    def stochastic_gradient_descent(self, X, Y, parameters, learning_rate, num_iterations, LAMBDA):
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
                #print("label", label)
                #print("data_in", data_in)

                AL, caches = self.forward_prop(data_in, parameters)
                loss = self.loss(AL, label, parameters)
                grads = self.backward_prop(AL, label, caches)
                parameters = self.update_parameters(parameters, grads, learning_rate, label.shape[0], LAMBDA)

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

        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")

        caches.append(cache)
        #print("shape",X.shape)
        assert(AL.shape == (1,X.shape[1]))

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
            A, activation_cache = utils.sigmoid(Z)

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
        Y - the labels of the training data; dimension (number of examples, 1)
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

        elif activation == "sigmoid":
            dZ = utils.sigmoid_backward(dA, activation_cache)
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

        m = Y.shape[0]

        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        loss = -1 * np.sum(logprobs) / m

        reg = 0
        L = len(parameters) // 2

        #for l in range(1, L + 1):
        #    w = parameters["W" + str(l)]
        #    print("w: ", w)
        #    print("w dot w.T", np.dot(w, w.T))
        #    reg += LAMBDA / float(2 * m) * np.dot(w, w.T)

        cost = np.squeeze(loss)

        return cost - reg

    def update_parameters(self, params, grads, learning_rate, m, LAMBDA):
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
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * (grads["dW" + str(l)] + LAMBDA / float(m) * params["W" + str(l)])
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
        AL, caches = self.forward_prop(X, self.parameters)
        AL = AL.reshape(Y.shape)
        predictions = self.classify(AL, 0.5)

        return predictions, Y

    def classify(self, outputs, threshold):
        """
        Arguments:
        outputs : a numpy array containing the set of scalars output by forward_prop
        threshold : the classification decision boundary

        Returns:
        outputs : a numpy array containing the class labels to which the outputs are mapped
        """
        outputs[outputs >= threshold] = 1
        outputs[outputs < threshold] = 0

        return outputs
