import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    hist = pd.read_csv("histogram.csv")
    # Filter normal and abnormal scores.v
    abn_scr = hist.loc[hist.labels == 1]['scores']
    nrm_scr = hist.loc[hist.labels == 0]['scores']

    # Create figure and plot the distribution.
    # fig, ax = plt.subplots(figsize=(4,4));
    sns.distplot(nrm_scr, label=r'Normal Scores')
    sns.distplot(abn_scr, label=r'Abnormal Scores')

    # statistische tests für scores einbinden Kolmogorow-Smirnow-Test sklearn.metrics https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
    if not os.path.isdir(os.path.join(os.getcwd(), "dataset")):
        print("test")

    plt.legend()
    plt.yticks([])
    plt.xlabel(r'Anomaly Scores')

    plt.show()