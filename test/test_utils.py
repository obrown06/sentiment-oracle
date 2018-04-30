import math
import sys
sys.path.insert(0, '../data/')
import numpy as np
import sklearn
import data_handler
from sklearn import metrics

def multiclass_accuracy(predictions, actual):
    predictions = np.asarray(predictions)
    actual = np.asarray(actual)

    correct = ((predictions == actual)).sum()
    incorrect = ((predictions != actual)).sum()
    near_correct = ((predictions == actual) | (predictions == actual + 1) | (predictions == actual - 1)).sum()
    near_incorrect = ((predictions != actual) & (predictions != actual + 1) & (predictions != actual - 1)).sum()

    correct_polarity = (((predictions < 3) & (actual < 3)) | ((predictions > 3) & (actual > 3)) | ((predictions == 3) & (actual == 3))).sum()
    incorrect_polarity = (((predictions < 3) & (actual >= 3)) | ((predictions > 3) & (actual <= 3)) | ((predictions == 3) & (actual != 3))).sum()

    return correct / (correct + incorrect), near_correct / (near_correct + near_incorrect), correct_polarity / (correct_polarity + incorrect_polarity)


def test_statistics(predictions, actual, pos_label):
    predictions = np.asarray(predictions)
    actual = np.asarray(actual)

    tp = ((predictions == pos_label) & (predictions == actual)).sum()
    fp = ((predictions == pos_label) & (predictions != actual)).sum()
    tn = ((predictions != pos_label) & (predictions == actual)).sum()
    fn = ((predictions != pos_label) & (predictions != actual)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fn + fp)

    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label)
    auc = metrics.auc(fpr, tpr)

    return precision, recall, specificity, accuracy, auc
