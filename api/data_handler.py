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
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle
import pre_process
from predict import predict


def clean(document):
    document_cleaner = pre_process.DocumentCleaner()
    [cleaned_document] = document_cleaner.clean([document])
    return cleaned_document

def invoke_predict(raw_json):

    # clean X_test
	json = json.loads(raw_json.decode())
    document = clean(document)
    model
	raw_model_output =  predict(model, model_usable_data)
	prediction = str(raw_model_output)
	return prediction #currently returns string
