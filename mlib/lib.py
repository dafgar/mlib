import matplotlib.pyplot as plt
import numpy as np


def plot_corr_heatmap(corr_dataframe):
    '''
    https://github.com/LeiG/Applied-Predictive-Modeling-with-Python.git
    '''
    import scipy.cluster.hierarchy as sch

    corr_matrix = np.array(corr_dataframe)
    col_names = corr_dataframe.columns

    Y = sch.linkage(corr_matrix, 'single', 'correlation')
    Z = sch.dendrogram(Y, color_threshold=0, no_plot=True)['leaves']
    corr_matrix = corr_matrix[Z, :]
    corr_matrix = corr_matrix[:, Z]
    col_names = col_names[Z]

    plt.imshow(corr_matrix, interpolation='nearest', aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.xticks(range(corr_matrix.shape[0]), col_names, rotation='vertical', fontsize=4)
    plt.yticks(range(corr_matrix.shape[0]), col_names[::-1], fontsize=4)

