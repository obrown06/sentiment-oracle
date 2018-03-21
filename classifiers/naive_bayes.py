import math
import numpy as np
import sklearn
from sklearn import metrics

class NaiveBayesBernoulliClassifier:

    def train(self, class_aggregated_docs):
        self.vocab_probs = dict()
        self.labels = set()
        self.priors = self.priors(class_aggregated_docs)

        for class_label, documents in class_aggregated_docs.items():
            num_docs_in_class = len(documents)
            self.labels.add(class_label)

            for document in documents:
                seen_words = set()
                for word in document.split(' '):
                    if word in seen_words:
                        continue

                    seen_words.add(word)

                    if word in self.vocab_probs:
                        self.vocab_probs[word][class_label] = self.vocab_probs[word][class_label] + 1 / float(num_docs_in_class + 2)
                    else:
                        class_probs = dict()
                        self.vocab_probs[word] = class_probs

                        for label in class_aggregated_docs:
                            class_probs[label] = 1 / float(num_docs_in_class + 2)

                        class_probs[class_label] = class_probs[class_label] + 1 / float(num_docs_in_class + 2)

    def predict(self, class_label, document, prob_sum):
        prob = math.log(self.priors[class_label]) + prob_sum
        doc_words = set()

        for word in document:
            if word not in doc_words and word in self.vocab_probs:
                prob = prob - math.log(1 - self.vocab_probs[word][class_label]) + math.log(self.vocab_probs[word][class_label])
                doc_words.add(word)

        return prob

    def prob_sum(self, class_label):
        prob_sum = 0

        for word in self.vocab_probs:
            prob_sum = prob_sum + math.log(1 - self.vocab_probs[word][class_label])

        return prob_sum

    def classify(self, document, prob_sum):
        max_prob = ""
        most_likely_class = ""

        for class_label in self.labels:
            prob = self.predict(class_label, document, prob_sum)

            if max_prob == "" or prob > max_prob:
                max_prob = prob
                most_likely_class = class_label

        return most_likely_class

    def priors(self, class_aggregated_docs):
        priors = dict()
        total = 0

        for class_label, documents in class_aggregated_docs.items():
            num_class_docs = len(documents)
            priors[class_label] = num_class_docs
            total = total + num_class_docs

        for class_label in class_aggregated_docs:
            priors[class_label] = priors[class_label] / float(total)

        return priors

    def test(self, class_aggregated_docs, pos_label):
        actual = []
        predictions = []

        for class_label, documents in class_aggregated_docs.items():
            prob_sum = self.prob_sum(class_label)

            for document in documents:
                prediction = self.classify(document.split(' '), prob_sum)
                actual.append(class_label)
                predictions.append(prediction)

        predictions = np.asarray(predictions)
        actual = np.asarray(actual)

        tp = ((predictions == pos_label) & (predictions == actual)).sum()
        fp = ((predictions == pos_label) & (predictions != actual)).sum()
        tn = ((predictions != pos_label) & (predictions == actual)).sum()
        fn = ((predictions != pos_label) & (predictions != actual)).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fn + fp)

        fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label = 1)
        auc = metrics.auc(fpr, tpr)

        return precision, recall, specificity, accuracy, auc

class NaiveBayesMultinomialClassifier:

    def train(self, classes_to_docs):
        self.labels = self.record_class_labels(classes_to_docs)
        self.priors = self.compute_priors(classes_to_docs)
        vocab_counts, self.class_totals = self.fill_vocab(classes_to_docs)
        self.vocab_probs = self.compute_vocab_probs(vocab_counts, self.class_totals)

    def fill_vocab(self, classes_to_docs):
        vocab = dict()
        totals = dict()

        for class_label, documents in classes_to_docs.items():
            totals[class_label] = 0
            self.fill_class_vocab(vocab, totals, class_label, documents)

        return vocab, totals

    def fill_class_vocab(self, vocab, totals, class_label, class_documents):
        for document in class_documents:
            for word in document.split(' '):
                totals[class_label] = totals[class_label] + 1
                self.add_word_to_class_vocab(vocab, class_label, word)

    def add_word_to_class_vocab(self, vocab, class_label, word):

        if word not in vocab:
            vocab[word] = self.create_empty_class_counts_dict()

        vocab[word][class_label] = vocab[word][class_label] + 1

    def create_empty_class_counts_dict(self):
        class_counts = dict()

        for label in self.labels:
            class_counts[label] = 0

        return class_counts

    def compute_vocab_probs(self, vocab_counts, class_totals):
        vocab_probs = dict()

        for class_label, class_total in class_totals.items():
            vocab_probs[class_label] = self.compute_class_vocab_probs(class_label, vocab_counts, class_total)

        return vocab_probs

    def compute_class_vocab_probs(self, class_label, vocab_counts, class_total):
        class_probs = dict()
        vocab_size = len(vocab_counts)

        for word in vocab_counts:
            class_count = vocab_counts[word][class_label]
            class_probs[word] = (class_count + 1) / float(vocab_size + class_total)

        return class_probs

    def record_class_labels(self, classes_to_docs):
        labels = set()

        for class_label in classes_to_docs:
            labels.add(class_label)

        return labels;

    def predict(self, class_label, document):
        class_total = self.class_totals[class_label]
        prob = math.log(self.priors[class_label])
        vocab_size = len(self.vocab_probs)
        class_vocab_probs = self.vocab_probs[class_label]

        for word in document:
            if word in class_vocab_probs:
                prob = prob + math.log(class_vocab_probs[word])

        return prob


    def classify(self, document):
        max_prob = ""
        most_likely_class = ""

        for class_label in self.labels:
            prob = self.predict(class_label, document)

            if max_prob == "" or prob > max_prob:
                max_prob = prob
                most_likely_class = class_label

        return most_likely_class

    def compute_priors(self, class_aggregated_docs):
        priors = dict()
        total = 0

        for class_label, documents in class_aggregated_docs.items():
            num_class_docs = len(documents)
            priors[class_label] = num_class_docs
            total = total + num_class_docs

        for class_label in class_aggregated_docs:
            priors[class_label] = priors[class_label] / float(total)

        return priors

    def test(self, class_aggregated_docs, pos_label):
        actual = []
        predictions = []

        for class_label, documents in class_aggregated_docs.items():
            for document in documents:
                prediction = self.classify(document.split(' '))
                actual.append(class_label)
                predictions.append(prediction)

        predictions = np.asarray(predictions)
        actual = np.asarray(actual)

        tp = ((predictions == pos_label) & (predictions == actual)).sum()
        fp = ((predictions == pos_label) & (predictions != actual)).sum()
        tn = ((predictions != pos_label) & (predictions == actual)).sum()
        fn = ((predictions != pos_label) & (predictions != actual)).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fn + fp)

        fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label = 1)
        auc = metrics.auc(fpr, tpr)

        return precision, recall, specificity, accuracy, auc
