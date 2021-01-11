import numpy as np
import sys
import json
import pandas as pd
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from difflib import SequenceMatcher


def depth(kmer):
    """
    Compute depth of kmer.
    Depth of kmer is len(kmer1)/2 + 1.
    For example:
         Depth (A) = 1
         Depth (TAT) = 3/2 + 1 = 2
         Depth (GTATG) = 5/2 + 1 = 3...
    """
    return int(len(kmer)/2 + 1)


def compute_topo_distance_between_kmers(kmer1, kmer2, mc_tree):
    """
    Naive Algorithm to find the Least Common Ancestor:
    Algorithm 1:
    Data: 1. Depth of kmer1 and kmer2
          2. Parent of each kmer.
          This information is already stored in the state dictionary.

    Algorithm:
    while depth(kmer1) != depth(kmer2):
        if depth(kmer1) > depth(kmer2):
            kmer2 <- parent(kmer2)
        elif depth(kmer2) > depth(kmer1):
            kmer1 <- parent(kmer1)
    # at this point, the depths of "kmer1" and "kmer2" should be the same.

    while kmer1 != kmer2:
        kmer1 <- parent(kmer1)
        kmer2 <- parent(kmer2)
    LCA <- kmer1
    return LCA

    Algorithm 2:
    Data: kmer1, kmer2 and LCA
    Distance(kmer1, kmer2) = Distance (kmer1, LCA) + Distance(kmer2, LCA)

    Parameters:
        kmer1 (str): kmer1
        kmer2 (str): kmer2
        mc_tree (dict) : mc_tree

    Returns:
        Distance(kmer1, kmer2)
    """
    node1 = kmer1
    node2 = kmer2

    while depth(node1) != depth(node2):
        if depth(node1) > depth(node2):
            node1 = mc_tree[node1]['parent']
        elif depth(node2) > depth(node1):
            node2 = mc_tree[node2]['parent']

    assert depth(node1) == depth(node2)

    while node1 != node2:
        node1 = mc_tree[node1]['parent']
        node2 = mc_tree[node2]['parent']

    assert node1 == node2
    lca = node1
    distance_between_nodes = (depth(kmer1) - depth(lca)) + (depth(kmer2) - depth(lca))
    return distance_between_nodes


def compute_longest_common_substring(kmer1, kmer2, mc_tree):
    match = SequenceMatcher(None, kmer1, kmer2).find_longest_match(0, len(
        kmer1), 0, len(kmer2))
    return match.size


def compute_distance_matrix(top_kmer_df, kmer_tree_dict, dist_fn):
    kmer_list = top_kmer_df['Kmer'].values

    distance_matrix_size = len(kmer_list)
    # Initialize the distance matrix
    distance_matrix = np.zeros(shape=(distance_matrix_size,
                                      distance_matrix_size))

    # For example, for a 1000 kmers, distance matrix will be an
    # matrix of zeros of size 1000 * 1000.
    counter_i = 0
    for kmer_i in kmer_list:
        counter_j = 0
        for kmer_j in kmer_list:
            # print(kmer_i, kmer_j, counter_i, counter_j)
            k_dist = dist_fn(kmer_i, kmer_j, mc_tree=kmer_tree_dict)
            distance_matrix[counter_i, counter_j] = k_dist
            counter_j += 1
        counter_i += 1
    print(distance_matrix)
    return distance_matrix


def main():
    tree_dictionary_fp = sys.argv[1]
    top_kmers_outfile = sys.argv[2]

    top_kmers = pd.read_csv(top_kmers_outfile, sep='\t',
                            header=0)

    with open(tree_dictionary_fp) as json_file:
        data_dictionary = json.load(json_file)

    dist_mat = compute_distance_matrix(top_kmer_df=top_kmers,
                                       kmer_tree_dict=data_dictionary,
                                       dist_fn=compute_longest_common_substring)




if __name__ == "__main__":
    main()