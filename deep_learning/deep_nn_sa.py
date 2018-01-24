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
            A, activation_cache = lrelu(Z)
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
            dZ = lrelu_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def lrelu(Z):
        A = math.max(Z, 0.01 * Z)
        return A, Z

    def sigmoid(Z):
        A = 1 / (1 + math.exp(-Z))
        return A, Z

    def sigmoid_backward(dA, Z):
        return np.multiply(dA, np.multiply(Z, 1 - Z))

    def lrelu_backward(dA, Z):
        return np.multiply(dA, np.where(Z > 0, Z, 0.01 * Z))

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
