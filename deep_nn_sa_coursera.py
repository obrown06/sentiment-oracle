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
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


        return parameters

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W, A) + b
        ### END CODE HERE ###

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
            ### END CODE HERE ###

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            ### END CODE HERE ###

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
            caches.append(cache)

            ### END CODE HERE ###

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
        caches.append(cache)

        ### END CODE HERE ###

        assert(AL.shape == (1,X.shape[1]))

        return AL, caches

    def loss(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[0]

        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
        cost = -1/m * np.sum(logprobs)
        ### END CODE HERE ###

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())

        return cost

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (≈ 3 lines of code)
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, keepdims = True, axis = 1)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

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
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        ### END CODE HERE ###

        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads


    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        ### END CODE HERE ###
        return parameters

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
dn.train(input_matrix_train, np.asarray(train_labels), layer_dims, 0.01, 50000, "batch")

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
