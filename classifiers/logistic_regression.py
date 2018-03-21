import numpy as np

class LogisticRegressionSAClassifier:

    LAMBDA = 1
    ALPHA = 2
    NITERATIONS = 2000

    def __init__(self, training_set, training_labels, NITERATIONS):
        self.training_set = training_set
        self.training_labels = training_labels
        self.features = {}
        self.build_feature_set(training_set)
        self.NITERATIONS = NITERATIONS

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, w, x):
        return self.sigmoid(np.dot(x.T, w))

    def classify(self, w, x):
        h = self.predict(w, x)

        for i in range(len(h)):
            if h[i] >= 0.5:
                h[i] = 1
            else:
                h[i] = 0

        return h

    def loss(self, x, w, y, h):
        m = x.shape[1]
        loss = -1 / float(m) * (np.dot(np.log(h).T, y) + np.dot(np.log(1 - h).T, 1 - y))
        reg = self.LAMBDA / float(2 * m) * np.dot(w.T, w)
        return loss - reg

    def grads(self, x, w, y, h):
        m = x.shape[1]
        grads = 1 / float(m) * np.dot(x, (h - y))

        reg = self.LAMBDA / float(m) * w
        reg[0] = 0

        return grads + reg

    def batch_gradient_descent(self, data):
        for i in range(self.NITERATIONS):
            h = self.predict(self.w, data)
            grads = self.grads(data, self.w, self.training_labels, h)
            self.w = self.w - self.ALPHA * grads

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", self.loss(data, self.w, self.training_labels, h))

        print("after training, weights are: ")
        print(self.w)

    def stochastic_gradient_descent(self, data):
        print("starting stochastic")
        m = data.shape[1]

        for i in range(self.NITERATIONS):
            for j in range(m):
                grads = (self.predict(self.w, data[:,j]) - self.training_labels[j]) * data[:,j]
                reg = self.LAMBDA * self.w / float(m)
                reg[0] = 0
                self.w = self.w - self.ALPHA * (grads + reg)

            if i % 100 == 0:
                print("loss after iteration ", i, " is: ", self.loss(data, self.w, self.training_labels, self.predict(self.w, data)))

        print("after training, weights are: ")
        print(self.w)

    def train(self, method):
        data_input = self.input_matrix(self.training_set)
        self.w = np.zeros(self.NFEATURES + 1)

        if method == "batch":
            self.batch_gradient_descent(data_input)
        else:
            self.stochastic_gradient_descent(data_input)

    def test(self, data, labels):
        data_input = self.input_matrix(data)
        c = self.classify(self.w, data_input)
        n_examples = len(labels)
        n_correct = 0

        for i in range(n_examples):
            if c[i] == labels[i]:
                n_correct = n_correct + 1

        return n_correct / float(n_examples)
