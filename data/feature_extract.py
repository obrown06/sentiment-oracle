import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def compute_ngrams(document, n):
    tokens = word_tokenize(document)

    if n == 1:
        return tokens
    else:
        return ngrams(tokens, n)

def build_feature_set(documents, NFEATURES, NGRAMS):
    feature_set = {}

    for n in range(NGRAMS):
        ngrams_list = []

        for i in range(len(documents)):
            document = documents[i]
            ngrams_list.extend(compute_ngrams(document, n + 1))

        fdist = FreqDist(ngram for ngram in ngrams_list)
        feature_set[n + 1] = [i[0] for i in fdist.most_common(NFEATURES // NGRAMS)]

    return feature_set

def doc_features(document, features, NGRAMS):
    all_grams_features = []

    for n in range(NGRAMS):
        doc_features = []
        ngrams = set(compute_ngrams(document, n + 1))

        for ngram in features[n + 1]:
            doc_features.append(1 if ngram in ngrams else 0)

        all_grams_features.extend(doc_features)

    return all_grams_features

def input_matrix(documents, features, NGRAMS):
    feature_set = []

    for document in documents:
        feature_set.append(doc_features(document, features, NGRAMS))

    return np.array(feature_set).T
