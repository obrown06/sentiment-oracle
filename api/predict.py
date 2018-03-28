# predict.py
# Script that should consist of a single method (predict) - passing data in a presumed parsimonious syntax to your model for prediction
import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle
import pre_process

sentiment_class_map = {0 : "negative", 1 : "positive"}

# take input pd data frame and return class
def predict(model, document):
	y_pred = model.classify(document)
	class_label = sentiment_class_map[y_pred]
	return class_label
