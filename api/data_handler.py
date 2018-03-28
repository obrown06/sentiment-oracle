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
import pre_process
import feature_extract
from predict import predict

classifiers_prefix = "../classifiers/"
data_prefix = "../data/"

def invoke_predict(raw_json):
    predictions = dict()
    post_data = json.loads(raw_json)
    cleaned_document = clean(post_data["document"])
    posted_names = post_data["classifiers"]
    used_names = set()

    for name in posted_names:
        if name in classifier_invoke_functions and name not in used_names:
            used_names.add(name)
            predictions[name] = classifier_invoke_functions[name](cleaned_document)

    return predictions

def invoke_predict_naive_bayes(document):
    model_filename = classifiers_prefix + 'nb_multinomial_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))
    model_usable_data = document.split(' ')
    return predict(model, model_usable_data)

def invoke_predict_logistic_regression(document):
    model_filename = classifiers_prefix + 'lr_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))

    extractor_filename = data_prefix + 'lr_extractor.p'
    extractor = pickle.load(open(extractor_filename, 'rb'))

    model_usable_data = extractor.extract_features_from_document(document, extractor.feature_set)
    ones = np.array([1])
    model_usable_data = np.concatenate((ones, model_usable_data), axis=0)

    return predict(model, model_usable_data)

def invoke_predict_deep_learning(document):
    model_filename = classifiers_prefix + 'nn_classifier.p'
    model = pickle.load(open(model_filename, 'rb'))

    extractor_filename = data_prefix + 'nn_extractor.p'
    extractor = pickle.load(open(extractor_filename, 'rb'))

    model_usable_data = extractor.extract_features_from_document(document, extractor.feature_set)
    model_usable_data = np.reshape(model_usable_data, (model_usable_data.shape[0], 1))

    return predict(model, model_usable_data)

classifier_invoke_functions = {"naive_bayes" : invoke_predict_naive_bayes,
                      "logistic_regression" : invoke_predict_logistic_regression,
                      "deep_learning" : invoke_predict_deep_learning}

def clean(document):
    document_cleaner = pre_process.DocumentCleaner()
    [cleaned_document] = document_cleaner.clean([document])
    return cleaned_document

#input = '{"document" : "I think she is a liar", "classifiers" : ["naive_bayes", "logistic_regression", "deep_learning"]}'
#print(invoke_predict(input))
#doc = "this is the worst i dislike"
#print(invoke_predict_logistic_regression(clean(doc)))
