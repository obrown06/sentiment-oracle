# falcon_gateway.py
import falcon
import json
import pickle
from data_handler import invoke_predict

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
        resp.body = ('\nThis is the PREDICT endpoint for Sentiment Analysis Classifier. \n'
        			 'Both requests and responses are served in JSON. \n'
        		     '\n'
        		     'INPUT:  JSON dict containing string for classification \n'
                     '        classification and list of zero or more names \n'
                     '        of requested classifiers. \n\n'
                     'Allowed classifier names are as follows: \n'
                     '        \'naive_bayes\' \n'
                     '        \'logistic_regression\' \n'
                     '        \'deep_learning\' \n\n'
                     'Example: \n'

                     '{"document": [string], "classifiers" : ["naive_bayes", "logistic_regression", ...]  \n\n'

        		     'OUTPUT: JSON dict mapping all unique POSTed \n'
                     '        classifier names to the classes returned \n'
                     '        by the associated classifier. \n\n'
                     'Example: \n'
                     '{"naive_bayes": [string], "logistic_regression": [string], ... }  \n\n')

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
        resp.body = json.dumps(invoke_predict(raw_json))

# falcon.API instances are callable WSGI apps. Never change this.
app = application = falcon.API()

# Resources are represented by long-lived class instances. Each Python class becomes a different "URL directory"
info = InfoResource()
predicts = PredictsResource()

# things will handle all requests to the '/things' URL path
app.add_route('/info', info)
app.add_route('/predicts', predicts)
