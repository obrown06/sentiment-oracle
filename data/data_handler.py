import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')

import clean
import csv
import naive_bayes
import math
import io
import sklearn
import pickle
import json
import glove_extractor
import bow_extractor
import numpy as np

PHRASE_COL_INDEX = 2
SENTIMENT_COL_INDEX = 3

def generate_nb_input(documents, labels, label_set):
    SHOULD_ADD_NEGATIONS = True
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, SHOULD_ADD_NEGATIONS)
    nb_input = partition_documents_by_class(documents, labels, label_set)

    return nb_input

def generate_bow_input(documents, extractor):
    SHOULD_ADD_NEGATIONS = True
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, SHOULD_ADD_NEGATIONS)
    bow_input = extractor.extract_features(documents)

    return bow_input

def generate_glove_input(documents, extractor):
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents)
    bow_input = extractor.extract_features(documents)

    return bow_input

def generate_bow_extractor(documents, nfeatures, ngrams):
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, True)
    #print("documents", documents)
    extractor = bow_extractor.BOWFeatureExtractor()
    extractor.build_feature_set(documents, nfeatures, ngrams)

    return extractor

def generate_glove_extractor(documents, nfeatures, ngrams):
    SHOULD_ADD_NEGATIONS = False
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, SHOULD_ADD_NEGATIONS)
    extractor = glove_extractor.GloveFeatureExtractor()
    extractor.build_feature_set(documents, nfeatures)

    return extractor

def generate_glove_embeddings(extractor, path_to_glove_embeddings, nfeatures, embed_size):
    feature_set = extractor.feature_set
    embeddings = extractor.extract_glove_embeddings(path_to_glove_embeddings, nfeatures, embed_size, feature_set)

    return embeddings

def load_balanced_data(n_samples_per_class, start_index, class_labels, path_to_data):
    documents = []
    labels = []
    class_counts = dict()

    for label in class_labels:
        class_counts[label] = 0

    end_index = 0

    with open(path_to_data, 'r', encoding='utf8') as file:
        reader = csv.reader(file, dialect="excel-tab")
        for line in reader:
            if reader.line_num == 1 or reader.line_num < start_index:
                continue
            document = line[PHRASE_COL_INDEX]
            label = int(line[SENTIMENT_COL_INDEX]) + 1

            if class_counts[label] < n_samples_per_class:
                documents.append(document)
                labels.append(label)
                class_counts[label] = class_counts[label] + 1

            if full(class_counts, n_samples_per_class):
                end_index = reader.line_num
                break

    return documents, labels, end_index

def load_data(n_samples, start_index, path_to_data):
    documents = []
    labels = []
    end_index = 0

    with open(path_to_data, 'r', encoding='utf8') as file:
        reader = csv.reader(file, dialect="excel-tab")
        for line in reader:

            if reader.line_num == 1 or reader.line_num < start_index:
                continue

            document = line[PHRASE_COL_INDEX]
            label = int(line[SENTIMENT_COL_INDEX]) + 1
            documents.append(document)
            labels.append(label)

            if reader.line_num == start_index + n_samples:
                end_index = reader.line_num
                break

    return documents, labels, end_index

def full(class_counts, required_count):

    for label, count in class_counts.items():
        if count < required_count:
            return False

    return True

def partition_documents_by_class(documents, labels, label_set):
    partitioned_documents = dict()

    for label in label_set:
        partitioned_documents[label] = subset(documents, labels, label)

    return partitioned_documents

def subset(documents, labels, label):
    subset = []
    for i in range(len(labels)):
        if labels[i] == label:
            subset.append(documents[i])

    return subset

def labels_to_indices(labels, label_set):
    indices = np.array([])

    for label in np.nditer(labels):
        indices = np.append(indices, label_set.index(label))

    return indices
