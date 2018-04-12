import numpy as np
import utils

class LogisticRegressionClassifier:

    def __init__(self, NITERATIONS = 2000, LAMBDA = 1, ALPHA = 0.01):
        self.NITERATIONS = NITERATIONS
        self.LAMBDA = LAMBDA
        self.ALPHA = ALPHA

    def train(self, X, Y, class_list, method):
        """
        Arguments:
        X : a numpy array of dimension [number of features] x [number of training examples], containing the
            values of every feature in every document in the training set
        Y : a numpy array of dimension [number of training examples] x 1, containing the class labels of
        every document in the training set
        class_list : a python list containing the labels of the possible output classes
        method : a string indicating the training method to be used

        Stores:

        self.w : a numpy array of dimension [number of training examples] containing the values of the weights
                 of the classifier (initialized to zero)
        """
        self.class_list = class_list

        ones = np.ones((1, X.shape[1]))
        X = np.concatenate((ones, X), axis=0)
        self.w = np.zeros((X.shape[0], len(class_list)))
        one_hot_Y = self.convert_to_one_hot(Y)

        if method == "batch":
            self.batch_gradient_descent(X, one_hot_Y)
        else:
            self.stochastic_gradient_descent(X, one_hot_Y)

    def convert_to_one_hot(self, Y):
        """
        Arguments:
        Y : a numpy array of dimension [number of documents] x 1, containing the class labels of
            a set of documents

        Stores:

        one_hot.w : a numpy array of dimension [number of documents] x 1, containing the "one-hot"
                    representation of all of the class labels for every document in the document set
        """
        one_hot = np.zeros((len(Y), len(self.class_list)))

        for i in range(len(Y)):
            one_hot[i, Y[i] - 1] = 1


        return one_hot

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
        print("X", X)
        print("Y", Y)
        print("self.w", self.w)
        for i in range(self.NITERATIONS):
            h = self.predict(X)
            grads_w = self.grads(X, Y, h)
            self.w = self.w - self.ALPHA * grads_w

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", self.loss(Y, h))

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
                print("loss after iteration ", i, " is: ", self.loss(Y, self.predict(X)))

    def loss(self, Y, h):
        """
        Arguments:
        Y - a "one-hot" representation of the labels of the training set; dimension (number of exmples, number of classes)
        h - a numpy array containing the "one-hot" predictions of the network; dimension (number of examples, number of classes)

        Returns:
        loss - a scalar; the cross entropy log loss, with added L2 norm
        """
        m = Y.shape[0]
        loss = - 1 / float(m) * np.sum(np.multiply(Y, np.log(h)))
        reg = 1 / float(2 * m) * self.LAMBDA * np.sum(np.multiply(self.w, self.w))
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
        h = np.array([])

        for i in range(X.shape[1]):
            doc_data = X[:,i].reshape(X[:,i].shape[0], 1)
            h = np.append(h, self.classify(doc_data))

        return h, Y

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
        return self.class_list[max]

    def predict(self, data):
        """
        Arguments:
        data     : a numpy array of dimension [number of features] x [number of examples], containing the
                   values of every feature in each document; includes a row of 1s which pair
                   with the w[0] (the bias)

        Returns:
        h : a numpy array of dimension [number of examples] x [number of classes] whose rows are "one-hot"
            arrays containing the predictions of the classifier
        """
        #print("data shape", data.shape)
        #print("data", data)
        #print("w", self.w)
        #print("w shape", self.w.shape)
        Z = np.dot(data.T, self.w)
        #print("Z", Z)
        #print("Z shape", Z.shape)
        h, cache = utils.softmax(Z)
        #print("h", h)
        return h
