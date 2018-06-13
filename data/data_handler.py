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
import keras_extractor
import bag_of_ngrams_extractor
import numpy as np

PHRASE_COL_INDEX = 2
SENTIMENT_COL_INDEX = 3

def generate_nb_input(documents, labels, label_set):
    SHOULD_ADD_NEGATIONS = True
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, SHOULD_ADD_NEGATIONS)
    nb_input = partition_documents_by_class(documents, labels, label_set)

    return nb_input

def generate_input(documents, extractor, SHOULD_ADD_NEGATIONS=True):
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, SHOULD_ADD_NEGATIONS)
    input = extractor.extract_features(documents)
    return input

def generate_bag_of_ngrams_extractor(documents, nfeatures, ngrams):
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, True)
    extractor = bag_of_ngrams_extractor.BagOfNGramsFeatureExtractor()
    extractor.build_feature_set(documents, nfeatures, ngrams)

    return extractor

def generate_keras_extractor(documents):
    cleaner = clean.DocumentCleaner()
    documents = cleaner.clean(documents, True)
    extractor = keras_extractor.KerasLSTMFeatureExtractor()
    extractor.build_feature_set(documents)

    return extractor

def generate_glove_extractor(documents, nfeatures):
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

def load_data(data_source, path_to_data, n_samples_train, n_samples_val, n_samples_test, class_labels, is_balanced=False):

    load_function = generate_load_function(data_source, is_balanced)
    test_documents, test_labels, test_end_index = load_function(n_samples_test, 0, path_to_data, class_labels)
    val_documents, val_labels, val_end_index = load_function(n_samples_test, test_end_index, path_to_data, class_labels)
    train_documents, train_labels, end_index = load_function(n_samples_train, val_end_index, path_to_data, class_labels)

    return train_documents, train_labels, val_documents, val_labels, test_documents, test_labels, end_index

def generate_load_function(data_source, data_is_balanced):

    if data_source == "ROTTEN_TOMATOES":
        if data_is_balanced:
            return load_balanced_rt_data
        else:
            return load_rt_data
    elif data_source == "YELP":
        if data_is_balanced:
            return load_balanced_yelp_data
        else:
            return load_yelp_data
    else:
        return load_amazon_data

def load_balanced_yelp_data(n_samples_per_class, start_index, path_to_data, class_labels):
    documents = []
    labels = []
    class_counts = dict()

    for label in class_labels:
        class_counts[label] = 0

    end_index = 0

    with open(path_to_data, 'r', encoding='utf8') as file:
        for i, line in enumerate(file):
            if i < start_index:
                continue

            data = json.loads(line)
            document = data["text"]
            label = data["stars"]

            if label in class_counts:
                if class_counts[label] < n_samples_per_class:
                    documents.append(document)
                    labels.append(label)
                    class_counts[label] = class_counts[label] + 1

                if full(class_counts, n_samples_per_class):
                    end_index = i
                    break

    return documents, labels, end_index

def load_yelp_data(n_samples, start_index, path_to_data, class_labels):
    documents = []
    labels = []
    end_index = 0

    with open(path_to_data, 'r', encoding='utf8') as file:
        for i, line in enumerate(file):
            if i < start_index:
                continue

            data = json.loads(line)
            document = data["text"]
            label = data["stars"]
            documents.append(document)
            labels.append(label)

            if i == start_index + n_samples:
                end_index = i
                break

    return documents, labels, end_index

def load_balanced_rt_data(n_samples_per_class, start_index, path_to_data, class_labels):
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

def load_rt_data(n_samples, start_index, path_to_data, class_labels=None):
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

def load_amazon_data(n_samples, start_index, path_to_data, class_labels):
    documents = []
    labels = []
    end_index = 0

    human_readable_names_to_labels = {"__label__1" : 1, "__label__2" : 2}

    with open(path_to_data, 'r', encoding='utf8') as file:
        for i, line in enumerate(file):
            if i < start_index:
                continue

            [name, document] = line.split(' ', 1)
            documents.append(document)
            labels.append(human_readable_names_to_labels[name])

            if i == start_index + n_samples:
                end_index = i
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
