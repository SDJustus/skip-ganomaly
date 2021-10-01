import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import os

import numpy as np

# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm,
                          target_names=["Normal", "Abnormal"],
                          cmap=None,
                          normalize=True,
                          save_path = None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    figure = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return figure


def plot_hist(file_name, threshold, title):
    hist = pd.read_csv(file_name)
    # Filter normal and abnormal scores.v
    abn_scr = hist.loc[hist.labels == 1]['scores']
    nrm_scr = hist.loc[hist.labels == 0]['scores']
    # Create figure and plot the distribution.
    # fig, ax = plt.subplots(figsize=(4,4));
    sns.distplot(nrm_scr, label=r'Normal Scores')
    sns.distplot(abn_scr, label=r'Abnormal Scores')
    plt.axvline(x=float(threshold), color="red", linestyle='--', label="Threshold")
    plt.legend()
    #plt.title(title)
    plt.yticks([])
    plt.xlabel(r'Anomaly Scores')
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.close()

def recall(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return (tn)/(tn+fp)

def precision(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return (tn)/(tn+fn)

def f1_score(conf_matrix):
    return (2*precision(conf_matrix)*recall(conf_matrix))/(precision(conf_matrix)+recall(conf_matrix))

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 18})
    cm = {"SSR": ([[64, 11], [1427, 8343]], "0.045")} #SSR
    cm["SSDD"] = ([[117, 27], [147, 620]], "0.151") #SSDD
    cm["VDD_c_128"] = ([[27, 8], [37, 132]], "0.165") #VDD_c_128
    cm["VDD_c_256"] = ([[29, 5], [26, 143]], "0.123") #VDD_c_256
    cm["VDD_bw_128"] = ([[24, 10], [48, 120]], "0.522") #VDD_bw_128
    cm["VDD_bw_256"] = ([[23, 11], [56, 112]], "0.307") #VDD_bw_256



    #plot_hist("histogram.csv")
    for key, value in cm.items():
        plot_confusion_matrix(cm=np.asarray(value[0]), target_names=["Normal", "Abnormal"], normalize=False, title="Confusion Matrix " + key)
        print(key + " recall: " + str(recall(np.asarray(value[0]))), ", precision: " + str(precision(np.asarray(value[0]))) + ", f1_score: " + str(f1_score(np.asarray(value[0]))))
        #plot_hist("histogram"+key+".csv", value[1], title="Histogramm " + key)