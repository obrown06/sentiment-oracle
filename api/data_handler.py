# data_handler.py
#
# Argument handler that does 4 things.
#
# 1. Decode: deserialize raw input from API POST request received in `falcon_gateway.py`
# 2. Preprocess: convert input data into form required for model, as specified in `predict.py`
# 3. Postprocess: convert prediction from model (from `predict.py`) into form that can be serializable for serving API response
# 4. Encode: serialize postprocessed data into valid JSON-esque format for API response, and pass back to `falcon_gateway.py`

import json
import sys
import numpy as np
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle
import clean
import bow_extractor
import glove_extractor
from predict import predict

pickle_prefix = "../pickle/"
data_prefix = "../data/"

def invoke_predict(raw_json):
    predictions = dict()
    post_data = json.loads(raw_json)
    document = post_data["document"]
    posted_names = post_data["classifiers"]
    used_names = set()

    for name in posted_names:
        if name in classifier_invoke_functions and name not in used_names:
            used_names.add(name)
            predictions[name] = classifier_invoke_functions[name](document)

    return predictions

def invoke_predict_naive_bayes(document):
    cleaned_document = clean_document_with_negations(document)
    model_filename = pickle_prefix + 'nb_multinomial_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))
    model_usable_data = cleaned_document.split(' ')
    return predict(model, model_usable_data)

def invoke_predict_logistic_regression(document):
    cleaned_document = clean_document_with_negations(document)
    model_filename = pickle_prefix + 'lr_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))
    extractor_filename = pickle_prefix + 'lr_extractor.p'
    extractor = pickle.load(open(extractor_filename, 'rb'))
    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    ones = np.array([1])
    model_usable_data = np.concatenate((ones, model_usable_data), axis=0)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

def invoke_predict_feed_forward(document):
    cleaned_document = clean_document_with_negations(document)
    model_filename = pickle_prefix + 'ff_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))

    extractor_filename = pickle_prefix + 'ff_extractor.p'
    extractor = pickle.load(open(extractor_filename, 'rb'))

    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

def invoke_predict_feed_forward_pt(document):
    cleaned_document = clean_document_with_negations(document)
    model_filename = pickle_prefix + 'pytorch_ff_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))

    extractor_filename = pickle_prefix + 'pytorch_ff_extractor.p'
    extractor = pickle.load(open(extractor_filename, 'rb'))

    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

def invoke_predict_lstm(document):
    cleaned_document = clean_document(document)
    model_filename = pickle_prefix + 'lstm_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))

    extractor_filename = pickle_prefix + 'pytorch_lstm_extractor.p'
    extractor = pickle.load(open(extractor_filename, 'rb'))

    model_usable_data = extractor.extract_features_from_document(cleaned_document)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

classifier_invoke_functions = {"naive_bayes" : invoke_predict_naive_bayes,
                               "logistic_regression" : invoke_predict_logistic_regression,
                               "feed_forward" : invoke_predict_feed_forward,
                               "feed_forward_pt" : invoke_predict_feed_forward_pt,
                               "lstm" : invoke_predict_lstm}

def clean_document(document):
    document_cleaner = clean.DocumentCleaner()
    [cleaned_document] = document_cleaner.clean([document])
    return cleaned_document

def clean_document_with_negations(document):
    CLEAN_WITH_NEGATIONS = True
    document_cleaner = clean.DocumentCleaner()
    [cleaned_document] = document_cleaner.clean([document], CLEAN_WITH_NEGATIONS)
    return cleaned_document

input = '{"document" : "It was mediocre", "classifiers" : ["naive_bayes", "logistic_regression", "feed_forward", "feed_forward_pt", "lstm"]}'
print(invoke_predict(input))
#doc = "this is the worst i dislike"
#print(invoke_predict_logistic_regression(clean(doc)))
