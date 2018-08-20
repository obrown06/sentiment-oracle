import json
import sys
import numpy as np
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle
import clean
import bag_of_ngrams_extractor
import lstm_keras
import glove_extractor
from predict import predict

pickle_prefix = "../pickle/"
data_prefix = "../data/"

def invoke_predict(raw_json, models, extractors):
    predictions = dict()
    post_data = json.loads(raw_json.decode(), encoding='utf-8')
    document = post_data["document"]
    posted_names = post_data["classifiers"]
    used_names = set()

    for name in posted_names:
        if name in classifier_invoke_functions and name not in used_names:
            used_names.add(name)
            predictions[name] = classifier_invoke_functions[name](document, models, extractors)

    return predictions

def invoke_predict_naive_bayes(document, models, extractors):
    cleaned_document = clean_document(document)
    model = models["nb"]
    model_usable_data = cleaned_document.split(' ')
    return predict(model, model_usable_data)

def invoke_predict_logistic_regression(document, models, extractors):
    cleaned_document = clean_document(document)
    model = models["lr"]
    extractor = extractors["lr"]
    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    ones = np.array([1])
    model_usable_data = np.concatenate((ones, model_usable_data), axis=0)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

def invoke_predict_feed_forward_gd(document, models, extractors):
    cleaned_document = clean_document(document)
    model = models["ff_gd"]
    extractor = extractors["ff_gd"]

    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

def invoke_predict_feed_forward_adam(document, models, extractors):
    cleaned_document = clean_document(document)
    model = models["ff_adam"]
    extractor = extractors["ff_adam"]

    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

def invoke_predict_lstm(document, models, extractors):
    cleaned_document = clean_document(document)
    model = models["lstm"]
    extractor = extractor = extractors["lstm"]

    model_usable_data = extractor.extract_features_from_document(cleaned_document)

    return predict(model, model_usable_data)

classifier_invoke_functions = {"naive_bayes" : invoke_predict_naive_bayes,
                               "feed_forward_gd" : invoke_predict_feed_forward_gd,
                               "feed_forward_adam" : invoke_predict_feed_forward_adam,
                               "lstm" : invoke_predict_lstm}

def clean_document(document):
    CLEAN_WITH_NEGATIONS = True
    document_cleaner = clean.DocumentCleaner()
    [cleaned_document] = document_cleaner.clean([document], CLEAN_WITH_NEGATIONS)
    return cleaned_document
