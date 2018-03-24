import math
import numpy as np
import sklearn
from sklearn import metrics

class NaiveBayesBernoulliClassifier:

    def train(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Stores:
            self.labels : a set containing labels for all classes
            self.priors : a set containing prior probabilities of observing each class
            self.vocab_probs : a dict of dicts containing the probabilities of observing each
                               word in each class, organized as : vocab[word][class_label]
        """
        self.labels = self.record_class_labels(classes_to_docs)
        self.priors = self.compute_priors(classes_to_docs)
        self.vocab_probs = self.compute_vocab_probs(classes_to_docs)

    def record_class_labels(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Returns:
        labels : a set containing the labels for all classes in the classes_to_docs dict.
        """

        labels = set()

        for class_label in classes_to_docs:
            labels.add(class_label)

        return labels;

    def compute_priors(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Returns:
        priors : a dictionary containing the prior probabilities of observing each
                 of the classes in the training set.
        """
        priors = dict()
        total = 0

        for class_label, documents in classes_to_docs.items():
            num_class_docs = len(documents)
            priors[class_label] = num_class_docs
            total = total + num_class_docs

        for class_label in classes_to_docs:
            priors[class_label] = priors[class_label] / float(total)

        return priors

    def compute_vocab_probs(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Returns: vocab_probs : a dict of dicts containing the probabilities of observing each
                           word in each class, organized as : vocab[word][class_label]
        """
        vocab_probs = dict()

        for class_label, documents in classes_to_docs.items():
            self.fill_class_vocab_probs(vocab_probs, class_label, documents)

        return vocab_probs

    def fill_class_vocab_probs(self, vocab_probs, class_label, documents):
        """
        Arguments:
        vocab_probs : a dict of dicts containing the probabilities of observing each
                           word in each class, organized as : vocab[word][class_label]
        class_label : the (integer) label of a class
        documents   : the training documents which are members of the class with the given [class_label]

        For each word in each document, adds a probability amount to the appropriate
        field in vocab_probs IFF it is the first time we have encountered the word in the document.
        """
        num_docs_in_class = len(documents)

        for document in documents:
            seen_words = set()
            for word in document.split(' '):
                if word not in seen_words:
                    seen_words.add(word)

                    if word not in vocab_probs:
                        vocab_probs[word] = self.create_empty_class_probs_dict(class_label, num_docs_in_class)

                    vocab_probs[word][class_label] = vocab_probs[word][class_label] + 1 / float(num_docs_in_class + 2)

    def create_empty_class_probs_dict(self, class_label, num_docs_in_class):
        """
        Arguments:
        class_label         : the (integer) label of a class
        num_docs_in_class   : the number of training documents which are members of the class with the given [class_label]

        Returns:
        class_probs : a dictionary containing the (equal) prior probabilities of
                      observing an as-yet-unobserved word in each class.
        """
        class_probs = dict()

        for label in self.labels:
            class_probs[label] = 1 / float(num_docs_in_class + 2)

        return class_probs

    def test(self, classes_to_test_docs):
        """
        Arguments:
        classes_to_test_docs : a python dict containing the set of test documents
                               segregated by class label.

        Returns:
        predictions : a list containing the most likely class labels for each document
                      in the test set (output by self.classify())
        actual      : a list containing the actual class labels for each document in the test set.
        """
        actual = []
        predictions = []

        for class_label, documents in classes_to_test_docs.items():
            default_prob = self.default_prob(class_label)

            for document in documents:
                prediction = self.classify(document.split(' '), default_prob)
                actual.append(class_label)
                predictions.append(prediction)

        return predictions, actual

    def classify(self, document, default_prob):
        """
        Arguments:
        document     : a document
        default_prob : the update to the prior probability of observing
                       the given class resulting from a document containing NO words in the
                       training vocabulary.

        Returns:
        most_likely_class_label : the class label in self.labels which results in the highest
                                  return value from predict()
        """
        max_prob = ""
        most_likely_class_label = ""

        for class_label in self.labels:
            prob = self.predict(class_label, document, default_prob)

            if max_prob == "" or prob > max_prob:
                max_prob = prob
                most_likely_class_label = class_label

        return most_likely_class_label

    def predict(self, class_label, document, default_prob):
        """
        Arguments:
        class_label  : the (integer) label of a class
        document     : a new document
        default_prob : the update to the prior probability of observing a given class
                       that we would compute IFF the given document contained NONE of
                       the words in the training vocabulary.

        Actions: Each time we encounter a new, distinct word in the given document
        which IS in the training vocabulary, we SUBTRACT the 'absentee' contribution of
        that word to default_prob and ADD the 'presentee' contribution of that word.

        We do it this way to avoid having to loop through the entire vocabulary every
        time we call predict -- this way, we get O(words in document) instead of
        O(words in vocabulary).

        Returns:
        prob : a proxy for the probability that the given document is a
               member of the given class (to be compared with  in the classify fn)
        """
        prob = math.log(self.priors[class_label]) + default_prob
        doc_words = set()

        for word in document:
            if word not in doc_words and word in self.vocab_probs:
                prob = prob - math.log(1 - self.vocab_probs[word][class_label]) + math.log(self.vocab_probs[word][class_label])
                doc_words.add(word)

        return prob

    def default_prob(self, class_label):
        """
        Arguments:
        class_label  : the (integer) label of a class

        Returns:
        default_prob : the update to the prior probability of observing
                       the given class resulting from a document containing NO words in the
                       training vocabulary.
        """

        default_prob = 0

        for word in self.vocab_probs:
            default_prob = default_prob + math.log(1 - self.vocab_probs[word][class_label])

        return default_prob

class NaiveBayesMultinomialClassifier:

    def train(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Stores:
            self.labels : a set containing labels for all classes
            self.priors : a set containing prior probabilities of observing each class
            self.vocab_probs : a dict of dicts containing the probabilities of observing each
                               word in each class, organized as : vocab[word][class_label]
        """
        self.labels = self.record_class_labels(classes_to_docs)
        self.priors = self.compute_priors(classes_to_docs)
        vocab_counts, self.class_vocab_totals = self.compute_vocab_counts(classes_to_docs)
        self.vocab_probs = self.compute_vocab_probs(vocab_counts, self.class_vocab_totals)

    def record_class_labels(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Returns:
        labels : a set containing the labels for all classes in the classes_to_docs dict.
        """
        labels = set()

        for class_label in classes_to_docs:
            labels.add(class_label)

        return labels;

    def compute_priors(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Returns:
        priors : a dictionary containing the prior probabilities of observing each
                 of the classes in the training set.
        """
        priors = dict()
        total = 0

        for class_label, documents in classes_to_docs.items():
            num_class_docs = len(documents)
            priors[class_label] = num_class_docs
            total = total + num_class_docs

        for class_label in classes_to_docs:
            priors[class_label] = priors[class_label] / float(total)

        return priors

    def compute_vocab_counts(self, classes_to_docs):
        """
        Arguments:
        classes_to_docs : a python dict containing the set of training documents
                          segregated by class label.

        Returns:
            vocab_counts : a dict of dicts containing the counts of each word in each class, organized
                           as vocab_counts[word][class_label]
            totals       : a dict containing the total number of words in each class
        """
        vocab_counts = dict()
        totals = dict()

        for class_label, documents in classes_to_docs.items():
            totals[class_label] = 0
            self.fill_class_vocab_counts(vocab_counts, totals, class_label, documents)

        return vocab_counts, totals

    def fill_class_vocab_counts(self, vocab_counts, totals, class_label, class_documents):
        """
        Arguments:
        vocab_counts : an empty dict, ultimately to contain the counts of each word
                       in each class, organized as vocab_counts[word][class_label]
        totals       : an empty dict ultimately to contain the total number of words in
                       each class
        class_label  : the label of one of the classes in the training set
        class_documents : the set of documents associated with the class with the given [class_label]
        """
        for document in class_documents:
            for word in document.split(' '):
                totals[class_label] = totals[class_label] + 1
                self.add_word_to_class_vocab(vocab_counts, class_label, word)

    def add_word_to_class_vocab(self, vocab_counts, class_label, word):
        """
        Arguments:
        vocab_counts : an empty dict, ultimately to contain the counts of each word
                       in each class, organized as vocab_counts[word][class_label]
        class_label  : the label of one of the classes in the training set
        word         : a word in one of the documents associated with the class with
                       the given [class_label]
        """
        if word not in vocab_counts:
            vocab_counts[word] = self.create_empty_class_counts_dict()

        vocab_counts[word][class_label] = vocab_counts[word][class_label] + 1

    def create_empty_class_counts_dict(self):
        """
        Returns:
        class_counts : a dictionary mapping each label in self.labels to an observation count initialized
                       to zero (to be modified as observations are made).
        """
        class_counts = dict()

        for label in self.labels:
            class_counts[label] = 0

        return class_counts

    def compute_vocab_probs(self, vocab_counts, class_totals):
        """
        Arguments:
        vocab_counts : a dict of dicts containing the counts of each word in each class, organized
                       as vocab_counts[word][class_label]
        totals       : a dict containing the total number of words in each class

        Returns:
            vocab_probs : a dict of dicts containing the probabilities of observing each word in each class,
                          organized as vocab_probs[class_label][word]
        """
        vocab_probs = dict()

        for class_label, class_total in class_totals.items():
            vocab_probs[class_label] = self.compute_class_vocab_probs(class_label, vocab_counts, class_total)

        return vocab_probs

    def compute_class_vocab_probs(self, class_label, vocab_counts, class_total):
        """
        Arguments:
        v : a dict of dicts containing the counts of each word in each class, organized
                       as vocab_counts[word][class_label]
        class_label  : the label of one of the classes in the training set
        class_total  : the total number of words in the training set belonging to the class with given [class_label]

        Returns:
        class_probs  : a dict containing the probabilities of observing each word in the given class,
        """
        class_probs = dict()
        vocab_size = len(vocab_counts)

        for word in vocab_counts:
            class_count = vocab_counts[word][class_label]
            class_probs[word] = (class_count + 1) / float(vocab_size + class_total)

        return class_probs

    def test(self, classes_to_docs):
        """
        Arguments:
        classes_to_test_docs : a python dict containing the set of test documents
                               segregated by class label.

        Returns:
        predictions : a list containing the most likely class labels for each document
                      in the test set (output by self.classify())
        actual      : a list containing the actual class labels for each document in the test set.
        """
        actual = []
        predictions = []

        for class_label, documents in classes_to_docs.items():
            for document in documents:
                prediction = self.classify(document.split(' '))
                actual.append(class_label)
                predictions.append(prediction)

        return predictions, actual

    def classify(self, document):
        """
        Arguments:
        document : a document whose class we want to determine

        Returns:
        most_likely_class_label : the class label in self.labels which results in the highest
                                  return value from predict()
        """
        max_prob = ""
        most_likely_class_label = ""

        for class_label in self.labels:
            prob = self.predict(class_label, document)

            if max_prob == "" or prob > max_prob:
                max_prob = prob
                most_likely_class_label = class_label

        return most_likely_class_label

    def predict(self, class_label, document):
        """
        Arguments:
        class_label  : the (integer) label of a class
        document     : a new document

        Returns:
        prob : a proxy for the probability that the given document is a
               member of the given class (to be compared with others in the self.classify())
        """
        class_total = self.class_vocab_totals[class_label]
        prob = math.log(self.priors[class_label])
        class_vocab_probs = self.vocab_probs[class_label]

        for word in document:
            if word in class_vocab_probs:
                prob = prob + math.log(class_vocab_probs[word])

        return prob
