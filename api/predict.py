import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../classifiers/')
import os
import pickle

def predict(model, document):
	pred = model.classify(document)
	return pred
