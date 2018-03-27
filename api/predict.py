# predict.py
# Script that should consist of a single method (predict) - passing data in a presumed parsimonious syntax to your model for prediction
import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle
import pre_process

# take input pd data frame and return dictionary with classificaiton
def predict_naive_bayes(X_test):

	# clean X_test
	doc_cleaner = pre_process.DocumentCleaner()
	[X_test] = doc_cleaner.clean([X_test])

	model = pickle.load(open("nb_multinomial_classifier.p", 'rb'))
	sentiment_class_map = {0 : "negative", 1 : "positive"}
	y_pred = model.classify(X_test)
	prediction_result = {'Sentiment': sentiment_class_map[y_pred]}
	return prediction_result

predict("Live it: Still have this works great my food is kept fresh inside the dome, so has a good seal to it.")
