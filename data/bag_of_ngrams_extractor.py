import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class BagOfNGramsFeatureExtractor:

    def build_feature_set(self, documents, NFEATURES, NGRAMS):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract
        NFEATURES   : an integer specifying the size of the feature set we would like to build
        NGRAMS      : an integer specifying the largest ngram value (= length of consecutive sequence of words)
                      we which we will use to construct features

        Returns:
        feature_set : a dict of dicts containing the set of most common ngrams in a sample, organized as
                      feature_set[ngram][word] = rank (by frequency) compared to other ngram features
        """
        feature_set = {}

        for n in range(NGRAMS):
            curr_grams = []

            for i in range(len(documents)):
                document = documents[i]
                curr_grams.extend(self.compute_ngrams(document, n + 1))

            fdist = FreqDist(ngram for ngram in curr_grams)
            most_common_ngrams = fdist.most_common(NFEATURES // NGRAMS)
            feature_set[n + 1] = {ngram_info[0] : rank for rank, ngram_info in enumerate(most_common_ngrams)}

        self.feature_set = feature_set

        return self.feature_set

    def extract_features(self, documents):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract
        feature_set : a dict of dicts containing the set of most common ngrams in a sample, organized as
                      feature_set[ngram][word] = rank (by frequency) compared to other ngram features

        Returns:
        features    : an np array containing the number of occurences of every feature in [feature_set] in
                      every document in [documents]
        """
        features = []

        for i in range(len(documents)):
            features.append(self.extract_features_from_document(documents[i]))

        features = np.array(features).T

        return features

    def extract_features_from_document(self, document):
        """
        Arguments:
        document    : a document whose features we would like to extract

        Returns:
        all_grams_features : an np array containing the frequencies of every feature in [feature_set]
                             in [document]
        """
        all_grams_features = np.array([])


        for n in range(len(self.feature_set)):
            ngrams_to_ranks = self.feature_set[n + 1]
            curr_gram_features = np.zeros(len(ngrams_to_ranks))
            ngrams = self.compute_ngrams(document, n + 1)

            for ngram in ngrams:
                if ngram in ngrams_to_ranks:
                    rank = ngrams_to_ranks[ngram]
                    curr_gram_features[rank] = curr_gram_features[rank] + 1

            all_grams_features = np.append(all_grams_features, curr_gram_features)

        return all_grams_features

    def compute_ngrams(self, document, n):
        """
        Arguments:
        document    : a document whose features we would like to extract
        n           : the length of the features we would like to look for

        Returns:
        ngrams      : a list of the ngrams features in the document
        """
        tokens = word_tokenize(document)

        if n == 1:
            return tokens
        else:
            return list(ngrams(tokens, n))
