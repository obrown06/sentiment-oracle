import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class FeatureExtractor:

    def extract_features(self, documents, feature_set, NGRAMS):
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
            features.append(self.extract_features_from_document(documents[i], feature_set, NGRAMS))

        features = np.array(features).T

        return features

    def extract_features_from_document(self, document, feature_set, NGRAMS):
        """
        Arguments:
        document    : a document whose features we would like to extract
        feature_set : a dict of dicts containing the set of most common ngrams in a sample, organized as
                      feature_set[ngram][word] = rank (by frequency) compared to other ngram features

        Returns:
        all_grams_features : an np array containing the frequencies of every feature in [feature_set]
                             in [document]
        """
        all_grams_features = np.array([])


        for n in range(NGRAMS):
            ngrams_to_ranks = feature_set[n + 1]
            curr_gram_features = np.zeros(len(ngrams_to_ranks))
            ngrams = self.compute_ngrams(document, n + 1)

            for ngram in ngrams:
                if ngram in ngrams_to_ranks:
                    rank = ngrams_to_ranks[ngram]
                    curr_gram_features[rank] = curr_gram_features[rank] + 1

            all_grams_features = np.append(all_grams_features, curr_gram_features)

        return all_grams_features

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
            ngrams = []

            for i in range(len(documents)):
                document = documents[i]
                ngrams.extend(self.compute_ngrams(document, n + 1))

            fdist = FreqDist(ngram for ngram in ngrams)
            most_common_ngrams = fdist.most_common(NFEATURES // NGRAMS)
            feature_set[n + 1] = {ngram_info[0] : rank for rank, ngram_info in enumerate(most_common_ngrams)}

        return feature_set

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
