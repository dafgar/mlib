from __future__ import division

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cluster_corr_matrix(corr_dataframe, linkage='single'):
    '''
    https://github.com/LeiG/Applied-Predictive-Modeling-with-Python.git
    '''
    import scipy.cluster.hierarchy as sch

    corr_matrix = np.array(corr_dataframe)
    col_names = corr_dataframe.columns

    Y = sch.linkage(corr_matrix, linkage, 'correlation')
    Z = sch.dendrogram(Y, color_threshold=0, no_plot=True)['leaves']
    corr_matrix = corr_matrix[Z, :]
    corr_matrix = corr_matrix[:, Z]
    col_names = col_names[Z]

    corr_dataframe = pd.DataFrame(corr_matrix, columns=col_names, index=col_names)

    return corr_dataframe


def check_degenerative(values):
    series = pd.Series(values)
    counts = series.value_counts()

    if len(counts) == 1:
        return True

    if len(counts) * 10 > len(series):
        return False

    if counts.iloc[0] / counts.iloc[1] > 20:
        return True

    return False

