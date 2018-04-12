import numpy as np
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

class GloveFeatureExtractor:

    def extract_ids(self, documents):
        """
        Arguments:
        documents   : a list of documents whose features we would like to extract

        Returns:
        ids    : a list of np arrays, each of which contains the ids of every token in a given document
        """
        ids = []

        for i in range(len(documents)):
            ids.append(self.document2ids(documents[i]))

        return ids

    def document2ids(self, document, unknown_id = 1):
        """
        Arguments:
        document    : a document whose features we would like to extract

        Returns:
        ids : a numpy array containing the ids (within self.token2id) of every token in the document
        """
        tokens = document.split()

        ids = []

        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id.get(token))
            else:
                ids.append(unknown_id)

        return np.array(ids)

    def create_token2id(self, documents, ntokens):
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

        id2token = ["PAD", "UNKNOWN"]

        for token_data in most_common_tokens_data:
            id2token.append(token_data[0])

        token2id = {token: id for id, token in enumerate(id2token)}
        self.token2id = token2id

        return self.token2id

    def initialize_embeddings(self, ntokens, embed_size):
        """ Creates a numpy array ntokens * embed_size, to be used for word
            embeddings initialized using a basic variant of Xavier
            initialization"""
        init_sd = 1 / np.sqrt(embed_size)
        embed_weights = np.random.normal(0, scale=init_sd, size=[ntokens, embed_size])
        embed_weights = embed_weights.astype(np.float32)

        return embed_weights

    def extract_glove_embeddings(self, embeddings_file, ntokens, embed_size, token2id):
        embeddings = self.initialize_embeddings(ntokens, embed_size)

        i = 0

        with open(embeddings_file, encoding="utf-8", mode="r") as file:
            for line in file:
                i = i + 1
                # Extract the word, and embeddings vector
                line = line.split()
                token, embed_vector = (line[0], np.array(line[1:], dtype=np.float32))

                # If it is in our vocab, then update the corresponding weights
                id = token2id.get(token)
                if id is not None:
                    embeddings[id] = embed_vector

                if i % 1000 == 0:
                    print("i is: ", i)

                if i == 10000:
                    break



        self.embeddings = embeddings

        return embeddings
