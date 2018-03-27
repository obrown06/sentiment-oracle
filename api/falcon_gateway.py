# falcon_gateway.py
import falcon
import os
import json
import pickle
from predict import invoke_predict

class InfoResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nThis is an API for a set of deployed sentiment analysis models  '
                     'which intake a text string and return its sentiment (positive or negative)\n'
                     'To learn more about this model, send a GET request to the /predicts endpoint.')


class PredictsResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nThis is the PREDICT endpoint for a Sentiment Analysis Classifier. \n'
        			 'Both requests and responses are served in JSON. \n'
        		     '\n'
        		     'INPUT:  Classifier and Document \n'
                     '     "classifier": string                 \n'
                     '     "document"  : string                 \n'
        		     'OUTPUT: Sentiment ["Positive"] // ["Negative"]   \n'
                     '   "Sentiment": [string]       \n\n')

    def on_post(self, req, resp):
        """Handles POST requests"""
        try:
            raw_json = req.stream.read()
        except Exception as ex:
            raise falcon.HTTPError(falcon.HTTP_400,
                'Error',
                ex.message)

        try:
            result_json = json.loads(raw_json.decode(), encoding='utf-8')
        except ValueError:
            raise falcon.HTTPError(falcon.HTTP_400,
                'Malformed JSON',
                'Could not decode the request body. The '
                'JSON was incorrect.')

        resp.status = falcon.HTTP_200
        resp.body = json.dumps(invoke_predict(model, raw_json))

# falcon.API instances are callable WSGI apps. Never change this.
app = falcon.API()

# Resources are represented by long-lived class instances. Each Python class becomes a different "URL directory"
info = InfoResource()
predicts = PredictsResource()

# things will handle all requests to the '/things' URL path
app.add_route('/info', info)
app.add_route('/predicts', predicts)
