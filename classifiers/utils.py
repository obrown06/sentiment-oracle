import numpy as np

def lrelu(Z):
    A = np.where(Z > 0, Z, 0.1 * Z)
    return A, Z

def relu(Z):
    A = np.where(Z > 0, Z, 0)
    return A, Z

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoid_backward(dA, Z):
    A, cache = sigmoid(Z)
    return np.multiply(dA, np.multiply(A, 1 - A))

def lrelu_backward(dA, Z):
    return np.multiply(dA, np.where(Z > 0, 1, 0.1))

def relu_backward(dA, Z):
    return np.multiply(dA, np.where(Z > 0, 1, 0))

def softmax(Z):
    max = np.amax(Z, axis=1)
    Z = Z - max.reshape((max.shape[0], 1))
    num = np.exp(Z)
    denom = np.sum(num, axis = 1)
    return num / denom.reshape((denom.shape[0], 1)), Z

def nn_softmax(Z):
    max = np.amax(Z, axis=0)
    Z = Z - max.reshape((1, max.shape[0]))
    num = np.exp(Z)
    denom = np.sum(num, axis = 0)
    return num / denom.reshape((1, denom.shape[0])), Z

def softmax_backward(dA, Z):

    return 0
