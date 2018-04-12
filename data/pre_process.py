import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

class DocumentCleaner:

    TERMINATORS = {',', '.', '!', ';', ':', '?', '\n'}
    NEGATIONS = {"not", "no", "never", "n't"}
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
        (r'ain\'t', 'am not'),
    ]

    def strip_labels_and_clean(self, documents, class_names):
        documents, labels = self.strip_labels(documents, class_names)
        cleaned_documents = self.clean(documents)
        return cleaned_documents, labels

    def strip_labels(self, documents, class_names):
        texts = []
        labels = []

        for d in documents:
            [class_name, text] = d.split(' ', 1)

            for i in range(len(class_names)):
                if class_name == class_names[i]:
                    labels.append(i)
                    texts.append(text)

        return texts, np.array(labels).T

    def clean(self, documents):
        for i in range(len(documents)):
            document = documents[i].lower()
            document = self.replace_neg_contractions(document);
            word_list = document.split(' ')
            self.remove_null_words(word_list)
            #self.add_negations(word_list)
            self.remove_terminators(word_list)
            #remove_stop_words(word_list)
            documents[i] = " ".join(word_list)

        return documents

    def replace_neg_contractions(self, document):
        for word in self.NEG_CONTRACTIONS:
            document = re.sub(word[0], word[1], document)

        return document;

    def add_negations(self, word_list):
        in_negation_zone = False
        for i in range(len(word_list)):
            word = word_list[i]
            if in_negation_zone:
                word_list[i] = "NOT_" + word
            if word in self.NEGATIONS or word[-3:] in self.NEGATIONS:
                in_negation_zone = not in_negation_zone
            if word[-1] in self.TERMINATORS:
                in_negation_zone = False

            return word_list

    def remove_terminators(self, word_list):
        for i in range(len(word_list)):
            word = word_list[i]
            last = len(word) - 1

            while last >= 0 and (word[last] in self.TERMINATORS or word[last] == '\n'):
                last = last - 1

            word_list[i] = word[0:last + 1]

        return word_list

    def remove_null_words(self, word_list):
        length = len(word_list)
        i = 0

        while i < length:
            if word_list[i] == "":
                del word_list[i]
                length = length - 1
            else:
                i = i + 1

            return word_list

    def remove_stop_words(self, word_list):
        length = len(word_list)
        i = 0

        while i < length:
            if word_list[i] in self.STOPWORDS:
                del word_list[i]
                length = length - 1
            else:
                i = i + 1

            return word_list
