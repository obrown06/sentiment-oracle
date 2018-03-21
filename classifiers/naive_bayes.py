import math

class NaiveBayesBernoulliClassifier:

    def train(self, class_aggregated_docs):
        print("in train")
        self.vocab_probs = dict()
        self.names = set()
        self.priors = self.priors(class_aggregated_docs)

        for class_name, documents in class_aggregated_docs.iteritems():
            num_docs_in_class = len(documents)
            self.names.add(class_name)

            for document in documents:
                seen_words = set()
                for word in document:
                    if word in seen_words:
                        continue

                    seen_words.add(word)

                    if word in self.vocab_probs:
                        self.vocab_probs[word][class_name] = self.vocab_probs[word][class_name] + 1 / float(num_docs_in_class + 2)
                    else:
                        class_probs = dict()
                        self.vocab_probs[word] = class_probs

                        for name in class_aggregated_docs:
                            class_probs[name] = 1 / float(num_docs_in_class + 2)

                        class_probs[class_name] = class_probs[class_name] + 1 / float(num_docs_in_class + 2)

    def predict(self, class_name, document, prob_sum):
        prob = math.log(self.priors[class_name]) + prob_sum
        doc_words = set()

        for word in document:
            if word not in doc_words and word in self.vocab_probs:
                prob = prob - math.log(1 - self.vocab_probs[word][class_name]) + math.log(self.vocab_probs[word][class_name])
                doc_words.add(word)

        return prob

    def prob_sum(self, class_name):
        prob_sum = 0

        for word in self.vocab_probs:
            prob_sum = prob_sum + math.log(1 - self.vocab_probs[word][class_name])

        return prob_sum


    def classify(self, document, prob_sum):
        max_prob = ""
        most_likely_class = ""

        for class_name in self.names:
            prob = self.predict(class_name, document, prob_sum)

            if max_prob == "" or prob > max_prob:
                max_prob = prob
                most_likely_class = class_name

        return most_likely_class

    def priors(self, class_aggregated_docs):
        priors = dict()
        total = 0

        for class_name, documents in class_aggregated_docs.iteritems():
            num_class_docs = len(documents)
            priors[class_name] = num_class_docs
            total = total + num_class_docs

        for class_name in class_aggregated_docs:
            priors[class_name] = priors[class_name] / float(total)

        return priors

    def test(self, class_aggregated_docs):
        print("testing")
        actual = []
        predictions = []
        mapping = dict()
        #un-hard code this
        mapping["__label__1"] = 0
        mapping["__label__2"] = 1

        for class_name, documents in class_aggregated_docs.iteritems():
            prob_sum = self.prob_sum(class_name)
            for document in documents:
                prediction = self.classify(document, prob_sum)
                actual.append(mapping[class_name])
                predictions.append(mapping[prediction])

        return actual, predictions

class NaiveBayesMultinomialClassifier:

    def vocab_with_counts(self, class_aggregated_docs):
        print("in vocab_with_counts")
        vocab = dict()
        totals = dict()

        for class_name, documents in class_aggregated_docs.iteritems():
            totals[class_name] = 0

            for document in documents:
                for word in document:
                    totals[class_name] = totals[class_name] + 1

                    if word in vocab:
                        class_counts = vocab[word]
                        class_counts[class_name] = class_counts[class_name] + 1
                    else:
                        class_counts = dict()
                        vocab[word] = class_counts

                        for name in class_aggregated_docs:
                            class_counts[name] = 0

                        class_counts[class_name] = class_counts[class_name] + 1

        return vocab, totals

    def train(self, class_aggregated_docs):
        print("in train")
        self.vocab_probs = dict()
        self.names = set()
        self.priors = self.priors(class_aggregated_docs)

        vocab, self.totals = self.vocab_with_counts(class_aggregated_docs)
        vocab_size = len(vocab)

        for class_name in class_aggregated_docs:
            print(class_name)
            self.names.add(class_name)
            class_total = self.totals[class_name]
            class_probs = dict()
            self.vocab_probs[class_name] = class_probs

            for word in vocab:
                class_count = vocab[word][class_name]
                class_probs[word] = (class_count + 1) / float(vocab_size + class_total)

    def predict(self, class_name, document):
        class_total = self.totals[class_name]
        prob = math.log(self.priors[class_name])
        vocab_size = len(self.vocab_probs)
        class_vocab_probs = self.vocab_probs[class_name]

        for word in document:
            if word in class_vocab_probs:
                prob = prob + math.log(class_vocab_probs[word])

        return prob


    def classify(self, document):
        max_prob = ""
        most_likely_class = ""

        for class_name in self.names:
            prob = self.predict(class_name, document)

            if max_prob == "" or prob > max_prob:
                max_prob = prob
                most_likely_class = class_name

        return most_likely_class

    def priors(self, class_aggregated_docs):
        priors = dict()
        total = 0

        for class_name, documents in class_aggregated_docs.iteritems():
            num_class_docs = len(documents)
            priors[class_name] = num_class_docs
            total = total + num_class_docs

        for class_name in class_aggregated_docs:
            priors[class_name] = priors[class_name] / float(total)

        return priors

    def test(self, class_aggregated_docs):
        print("testing")
        actual = []
        predictions = []
        mapping = dict()
        mapping["__label__1"] = 0
        mapping["__label__2"] = 1

        for class_name, documents in class_aggregated_docs.iteritems():
            for document in documents:
                prediction = self.classify(document)

                actual.append(mapping[class_name])
                predictions.append(mapping[prediction])

        return actual, predictions
