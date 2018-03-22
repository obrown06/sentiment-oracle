import math
import numpy as np
import sklearn
from sklearn import metrics


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

    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label = 1)
    auc = metrics.auc(fpr, tpr)

    return precision, recall, specificity, accuracy, auc
