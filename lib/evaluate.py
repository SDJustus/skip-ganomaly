""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function
from collections import OrderedDict
import matplotlib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
matplotlib.use('Agg')
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, fbeta_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import json
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
    fpr, tpr, t = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    #threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = roc_t['threshold']
    threshold = list(threshold)[0]
    #print(list(threshold))
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

    return roc_auc, threshold, t

def auprc(labels, scores):
    ap = average_precision_score(labels, scores, pos_label=1)
    return ap

def get_values_for_pr_curve(y_trues, y_preds, thresholds):
    precisions = []
    recalls = []
    tn_counts = []
    fp_counts = []
    fn_counts = []
    tp_counts = []
    for threshold in thresholds:
        y_preds_new = [1 if ele >= threshold else 0 for ele in y_preds] 
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds_new).ravel()
        if len(set(y_preds_new)) == 1:
            print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
            continue
        
        precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
        precisions.append(precision)
        recalls.append(recall)
        tn_counts.append(tn)
        fp_counts.append(fp)
        fn_counts.append(fn)
        tp_counts.append(tp)
        
        
    
    return np.array(tp_counts), np.array(fp_counts), np.array(tn_counts), np.array(fn_counts), np.array(precisions), np.array(recalls), len(thresholds)

def get_performance(y_trues, y_preds):
    fpr, tpr, t = roc_curve(y_trues, y_preds)
    roc_score = auc(fpr, tpr)
    ap = average_precision_score(y_trues, y_preds, pos_label=1)
    recall_dict = dict()
    precisions = [0.996, 0.99, 0.95, 0.9]
    temp_dict=dict()
    
    for th in t:
        y_preds_new = [1 if ele >= th else 0 for ele in y_preds] 
        if len(set(y_preds_new)) == 1:
            print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
            continue
        
        precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
        temp_dict[str(precision)] = recall
    p_dict = OrderedDict(sorted(temp_dict.items(), reverse=True))
    for p in precisions:   
        for precision, recall in p_dict.items(): 
            if precision<=p:
                print(f"writing {p}; {precision}")
                recall_dict["recall at pr="+str(p)+"(real_value="+str(precision)+")"] = recall
                break
            else:
                continue
    
    
    #Threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = roc_t['threshold']
    threshold = list(threshold)[0]
    
    
    
    y_preds = [1 if ele >= threshold else 0 for ele in y_preds] 
    
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds, average="binary", pos_label=1)
    f05_score = fbeta_score(y_trues, y_preds, beta=0.5, average="binary", pos_label=1)
    #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
    conf_matrix = confusion_matrix(y_trues, y_preds)
    performance = OrderedDict([ ('auc', roc_score), ("ap", ap), ('precision', precision),
                                ("recall", recall), ("f1_score", f1_score), ("f05_score", f05_score), ("conf_matrix", conf_matrix),
                                ("threshold", threshold)])
    performance.update(recall_dict)
                                
    return performance, t, y_preds

def write_inference_result(file_names, y_preds, y_trues, outf):
    classification_result = {"tp": [], "fp": [], "tn": [], "fn": []}
    for file_name, gt, anomaly_score in zip(file_names, y_trues, y_preds):
        anomaly_score=int(anomaly_score)
        if gt == anomaly_score == 0:
            classification_result["tp"].append(file_name)
        if anomaly_score == 0 and gt != anomaly_score:
            classification_result["fp"].append(file_name)
        if gt == anomaly_score == 1:
            classification_result["tn"].append(file_name)
        if anomaly_score == 1 and gt != anomaly_score:
            classification_result["fn"].append(file_name)
                
    with open(outf, "w") as file:
        json.dump(classification_result, file, indent=4)