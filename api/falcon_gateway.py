# falcon_gateway.py
import falcon
import sys
sys.path.insert(0, '../classifiers')
import lstm_keras
import json
import pickle
from data_handler import invoke_predict
from falcon_cors import CORS

cors = CORS(allow_origins_list=['https://nicholasbrown.io/sentiment-oracle', 'http://nicholasbrown.io/sentiment-oracle', 'http://localhost:8000', 'http://localhost:8000/software'])
public_cors = CORS(allow_all_origins=True)

models = {"nb": pickle.load(open("../pickle/nb_multinomial_classifier.p", "rb")),  "ff_gd" : pickle.load(open("../pickle/ff_classifier.p", "rb")), "ff_adam" : pickle.load(open("../pickle/pytorch_ff_classifier.p", "rb")), "lstm" : lstm_keras.load_keras("../pickle/keras_lstm_classifier.h5", "../pickle/keras_lstm_wrapper.p")}

extractors = {"ff_gd" : pickle.load(open("../pickle/ff_extractor.p", "rb")), "ff_adam" : pickle.load(open("../pickle/pytorch_ff_extractor.p", "rb")), "lstm" : pickle.load(open("../pickle/keras_lstm_extractor.p", "rb"))}

class InfoResource(object):
    cors = public_cors
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nThis is an API for Nick\'s collection \n'
                     'of deployed sentiment classifiers. \n'
                     'Each of these intake a text string and return \n'
                     'the sentiment (on a 1 to 5 scale) which \n'
                     'it expresses. To learn more about this model, \n'
                     'send a GET request to the /predicts endpoint.')


class PredictsResource(object):
    cors = public_cors
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.body = ('\nThis is the PREDICTS endpoint for Nick\'s collection of \n'
                     'sentiment classifiers. Both requests and responses are served in JSON. \n\n'

                     'The classifiers available for use are as follows: \n\n'
                     ' 1 :       \'naive_bayes\' \n\n'
                     ' 2 :       \'feed_forward_gd\' \n\n'
                     ' 3 :       \'feed_forward_adam\' \n\n'
                     ' 4 :       \'lstm\' \n\n'

                     'Each classifier intakes a string and outputs a sentiment class label from 1 (most negative) \n'
                     'to 5 (most positive) \n\n'

        		     'JSON REQUEST:  JSON dict containing 1) string for classification \n'
                     '               classification and 2) list of zero or more names \n'
                     '               of requested classifiers. \n\n'

                     'JSON RESPONSE: JSON dict containing all unique POSTed names which \n'
                     '               are also valid classifier names, along with the classes \n'
                     '               returned by the associated classifiers. \n\n'

                     'You can POST to this endpoint from terminal using the following command:\n\n'
                     'curl -H "Content-Type: application/json" -X POST -d \'{\"document\": \"STRING WHICH YOU WISH TO CLASSIFY\", \"classifiers\" : [\"name1\", \"name2\"] }\' http://api.nlp-sentiment.com/predicts \n\n'

                     'Example: \n\n'

                     'JSON REQUEST:\n'
                     '{"document": "I hate this", "classifiers" : {"naive_bayes", "feed_forward_gd"}  \n\n'
                     'JSON RESPONSE: \n'
                     '{"naive_bayes": "1", "feed_forward_gd" : "1"}  \n\n')

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
        resp.body = json.dumps(invoke_predict(raw_json, models, extractors))

# falcon.API instances are callable WSGI apps. Never change this.
app = application = falcon.API(middleware=[cors.middleware])

# Resources are represented by long-lived class instances. Each Python class becomes a different "URL directory"
info = InfoResource()
predicts = PredictsResource()

# things will handle all requests to the '/things' URL path
app.add_route('/info', info)
app.add_route('/predicts', predicts)
