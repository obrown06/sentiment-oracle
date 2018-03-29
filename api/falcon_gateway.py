# falcon_gateway.py
import falcon
import json
import pickle
from data_handler import invoke_predict

class InfoResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nThis is an API for Nick\'s collection \n'
                     'of deployed sentiment classifiers. \n'
                     'Each of these intake a text string and return \n'
                     'the sentiment ("positive" or "negative") which \n'
                     'it expresses. To learn more about this model, \n'
                     'send a GET request to the /predicts endpoint.')


class PredictsResource(object):
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nThis is the PREDICT endpoint for Nick\'s collection of \n'
                     'sentiment classifiers. Both requests and responses are served in JSON. \n\n'
        		     'INPUT:  JSON dict containing 1) string for classification \n'
                     '        classification and 2) list of zero or more names \n'
                     '        of requested classifiers. \n\n'
                     'Allowed classifier names are as follows: \n\n'
                     ' 1 :       \'naive_bayes\' \n'
                     ' 2 :       \'logistic_regression\' \n'
                     ' 3 :       \'deep_learning\' \n\n'

                     'Each classifier outputs one of two sentiment class names: \n\n'
                     ' 1 :       \'negative\'   \n'
                     ' 2 :       \'positive\'   \n\n'

                     'Example: \n'

                     '{"document": "I hate this", "classifiers" : {"naive_bayes", "logistic_regression"}  \n\n'

        		     'OUTPUT: JSON dict mapping all unique POSTed \n'
                     '        classifier names to the classes returned \n'
                     '        by the associated classifier. \n\n'
                     'Example: \n'
                     '{"naive_bayes": "negative", "logistic_regression": "negative"}  \n\n')

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
