# predict.py
# Script that should consist of a single method (predict) - passing data in a presumed parsimonious syntax to your model for prediction
import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle

# take input pd data frame and return class

def predict(model, document):
	pred = model.classify(document)
	return pred
