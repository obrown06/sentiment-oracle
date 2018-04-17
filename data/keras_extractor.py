import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from keras.preprocessing import sequence
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

MAX_WORD_COUNT = 60

class KerasLSTMFeatureExtractor:

    def __init__(self):
        self.tokenizer = Tokenizer()

    def build_feature_set(self, documents):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract

        Returns:
        feature_set : a dict of dicts containing the set of most common ngrams in a sample, organized as
                      feature_set[ngram][word] = rank (by frequency) compared to other ngram features
        """
        documents = np.array(documents)
        self.tokenizer.fit_on_texts(documents)
        return self.tokenizer

    def extract_features(self, documents):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract

        Returns:
        features    : an np array containing the number of occurences of every feature in [feature_set] in
                      every document in [documents]
        """
        documents = np.array(documents)
        sequences = self.tokenizer.texts_to_sequences(documents)
        padded_sequences = sequence.pad_sequences(sequences, MAX_WORD_COUNT)

        return padded_sequences

    def extract_features_from_document(self, document):
        """
        Arguments:
        document    : a document whose features we would like to extract

        """
        return self.extract_features([document])

    def vocab_size(self):

        return len(self.tokenizer.word_index) + 1
