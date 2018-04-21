import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class GloveFeatureExtractor:

    def extract_features(self, documents):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract

        Returns:
        ids    : a list of np arrays, each of which contains the ids of every token in a given document
        """
        features = []

        for i in range(len(documents)):
            features.append(self.extract_features_from_document(documents[i]))

        return features

    def extract_features_from_document(self, document, unknown_id = 1):
        """
        Arguments:
        document    : a document whose features we would like to extract

        Returns:
        ids : a numpy array containing the ids (within self.feature_set) of every token in the document
        """
        tokens = document.split()

        features = []

        for token in tokens:
            if token in self.feature_set:
                features.append(self.feature_set.get(token))
            else:
                features.append(unknown_id)

        features = np.array(features)

        return features

    def build_feature_set(self, documents, ntokens):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract
        NFEATURES   : an integer specifying the size of the feature set we would like to build

        Returns:
        feature_set : a dict of dicts containing the set of most common ngrams in a sample, organized as
                      feature_set[ngram][word] = rank (by frequency) compared to other ngram features
        """
        tokens = []
        for i in range(len(documents)):
            document = documents[i]
            tokens.extend(word_tokenize(document))

        fdist = FreqDist(token for token in tokens)
        most_common_tokens_data = fdist.most_common(ntokens - 2)

        id2token = ["<PAD>", "<UNKNOWN>"]

        for token_data in most_common_tokens_data:
            id2token.append(token_data[0])

        feature_set = {token: id for id, token in enumerate(id2token)}
        self.feature_set = feature_set

        return self.feature_set

    def initialize_embeddings(self, ntokens, embed_size):
        """ Creates a numpy array ntokens * embed_size, to be used for word
            embeddings initialized using a basic variant of Xavier
            initialization"""
        init_sd = 1 / np.sqrt(embed_size)
        embed_weights = np.random.normal(0, scale=init_sd, size=[ntokens, embed_size])
        embed_weights = embed_weights.astype(np.float32)

        return embed_weights

    def extract_glove_embeddings(self, embeddings_file, ntokens, embed_size, feature_set):
        embeddings = self.initialize_embeddings(ntokens, embed_size)

        i = 0
        with open(embeddings_file, encoding="utf-8", mode="r") as file:
            for line in file:
                i = i + 1
                # Extract the word, and embeddings vector
                line = line.split()
                token, embed_vector = (line[0], np.array(line[1:], dtype=np.float32))

                # If it is in our feature_set, then update the corresponding weights
                id = feature_set.get(token)
                if id is not None:
                    embeddings[id] = embed_vector
                if i == 100000:
                    break




        self.embeddings = embeddings

        return embeddings
