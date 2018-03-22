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
