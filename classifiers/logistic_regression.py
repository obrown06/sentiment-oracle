import numpy as np
import utils

class LogisticRegressionClassifier:

    def __init__(self, NITERATIONS = 2000, LAMBDA = 1, ALPHA = 0.2):
        self.NITERATIONS = NITERATIONS
        self.LAMBDA = LAMBDA
        self.ALPHA = ALPHA

    def train(self, X, Y, method):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
        every document in the training set
        method : a string indicating the training method to be used

        Stores:

        self.w : a numpy array of dimension [number of training examples] containing the values of the weights
                 of the classifier (initialized to zero)
        """
        ones = np.ones((1, X.shape[1]))
        X = np.concatenate((ones, X), axis=0)

        self.w = np.zeros(X.shape[0])

        if method == "batch":
            self.batch_gradient_descent(X, Y)
        else:
            self.stochastic_gradient_descent(X, Y)

    def batch_gradient_descent(self, X, Y):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
            every document in the training set

        Stores:
        self.w : a numpy array of dimension [number of features] x 1, containing the updated values of the
                 weights after a batch update (performed once per iteration)
        """
        for i in range(self.NITERATIONS):
            h = self.predict(X)
            grads_w = self.grads(X, Y, h)
            self.w = self.w - self.ALPHA * grads_w

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", self.loss(X, Y, h))

    def grads(self, X, Y, h):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
            every document in the training set
        h : a numpy array of dimension [number of training examples] x 1, containing the class labels predicted
            by the classifier for each document in the training set

        Returns:
        grads : a numpy array of dimension [number of features] x 1, containing the gradients to be added to each
                weight in the classifier. Computed by combining the simple gradients (grads) with a regularization
                term (reg).
        """
        m = X.shape[1]
        grads = 1 / float(m) * np.dot(X, (h - Y))
        reg = self.LAMBDA * self.w / float(m)
        reg[0] = 0
        grads = grads + reg

        return grads

    def stochastic_gradient_descent(self, X, Y):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
            every document in the training set

        Stores:
        self.w : a numpy array of dimension [number of features] x 1, containing the updated values of the
                 weights after a stochastic update (performed once per training example)
        """
        m = X.shape[1]

        for i in range(self.NITERATIONS):
            for j in range(m):
                h = self.predict(X[:,j])
                grads = (h - Y[j]) * X[:,j]
                reg = 2 * self.LAMBDA * self.w / float(m)
                self.w = self.w - self.ALPHA * (grads + reg)

            if i % 100 == 0:
                #print("X", X)
                #print("w", self.w)
                #print("X dot w", np.dot(X.T, self.w))
                #print("sigmoid()", utils.sigmoid(np.dot(X.T, self.w)))
                print("loss after iteration ", i, " is: ", self.loss(X, Y, self.predict(X)))

    def loss(self, X, Y, h):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y - the actual labels of the training set; dimension (number of exmples, 1)
        h - a numpy array containing the predictions of the network; dimension (number of examples, 1)

        Returns:
        loss - a scalar; the cross entropy log loss, with added L2 norm
        """
        m = X.shape[1]
        loss = -1 / float(m) * (np.dot(np.log(h).T, Y) + np.dot(np.log(1 - h).T, 1 - Y))
        reg = 1 / float(2 * m) * self.LAMBDA * np.dot(self.w.T, self.w)
        loss = loss + reg
        return loss

    def test(self, X, Y):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of test examples], containing the
            values of every feature in every document in the test set; includes a row of 1s which pair
            with the w[0] (the bias)
        Y - the actual labels of the test set; dimension (number of exmples, 1)

        Returns:
        h : a numpy array containing the predicted class labels for each document
            in the test set (output by self.classify())
        actual : a numpy array containing the actual class labels for each document in the test set.
        """
        ones = np.ones((1, X.shape[1]))
        X = np.concatenate((ones, X), axis=0)

        h = self.classify(X)
        return h, Y

    def classify(self, X):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of examples], containing the
            values of every feature in every document in the given set; includes a row of 1s which pair
            with the w[0] (the bias)

        Returns:
        h : a numpy array containing the predicted class labels for each document
            in the test set.
        """
        h = self.predict(X)

        for i in range(len(h)):
            if h[i] >= 0.5:
                h[i] = 1
            else:
                h[i] = 0

        return h

    def predict(self, X):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of examples], containing the
            values of every feature in every document in the given set; includes a row of 1s which pair
            with the w[0] (the bias)

        Returns:
        A : the outputs of the classifier applied to the given set of documents
        """
        A, cache = utils.sigmoid(np.dot(X.T, self.w))
        return A
