""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)

def evaluate(labels, scores, metric='roc', output_directory="./", epoch=0):
    if metric == 'roc':
        return roc(labels, scores, output_directory=output_directory, epoch=epoch)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, saveto=True, output_directory="./", epoch = 0):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()
    #labels = labels - 1
    #print(labels)
    # True/False Positive Rates.
    fpr, tpr, t = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    #threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    print(roc_t['threshold'])
    threshold = roc_t['threshold']
    threshold = list(threshold)[0]
    print(list(threshold))
    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(output_directory + "/ROC" + str(epoch) + ".png")
        plt.close()

    return roc_auc, threshold

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap