import numpy as np
import pandas as pd
import sys
from sklearn.cluster import AgglomerativeClustering
import compute_distances
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
import matplotlib
from scipy.cluster.hierarchy import fcluster

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def cluster(lcs_mat):
    # Convert the LCS based distance matrix into condensed vector form
    # Documentation on the format of the condensed vector form is here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
    # Note: The scipy linkage function requires a condensed distance matrix.

    cond_dist_mat = squareform(X=lcs_mat, force='no')
    # print(cond_dist_mat)
    # print(lcs_mat.shape)
    # print(len(cond_dist_mat))
    ag = linkage(y=cond_dist_mat, method='complete')
    # Note: The metric option in linkage is ignored if a cond. distance
    # matrix is passed into it!
    # print(ag)
    # dendrogram(Z=ag, p=10, truncate_mode='level')
    # plt.savefig('test.dg.png')
    cluster_labels = fcluster(Z=ag, t=8.5, criterion='distance')
    return cluster_labels


def summarize_clusters(kmer_df):
    print(kmer_df)
    kmer_df_c0 = kmer_df[kmer_df['cluster_label'] == 5]
    print(kmer_df_c0)



