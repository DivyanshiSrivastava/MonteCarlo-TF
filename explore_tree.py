import numpy as np
import sys
import json
import itertools
import mcts
import pandas as pd


def explore(data_dict, top_n, outfile):
    kmer_list = []
    score_list = []

    for kmer_key in data_dict:
        kmer_list.append(kmer_key)
        try:
            score = data_dict[kmer_key]['reward']/data_dict[kmer_key]['visits']
        except KeyError:
            score = 0
        score_list.append(score)

    # sort order
    sort_order = np.argsort(score_list)[::-1]
    # sort the k-mer list based on scores
    sorted_kmer_list = np.array(kmer_list)[sort_order]
    sorted_score_list = np.array(score_list)[sort_order]

    rank = 0
    ranked_list = []
    for k, score in zip(sorted_kmer_list, sorted_score_list):
        ranked_list.append((k, score))
        rank += 1
        if rank >= top_n:
            data_for_file = pd.DataFrame(ranked_list,
                                         columns=['Kmer', 'Score'])
            print(data_for_file)
            data_for_file.to_csv(outfile, index=False, sep='\t')
            break



def traverse_tree(node, data_dict, score_list, node_list):
    print(node)
    if data_dict[node]['status'] == 'unvisited':
        print('reached leaf node, completed iteration')
        print(node_list)
        print(score_list)
    else:
        try:
            child_nodes = mcts.Tree.get_node_children(node=node)
            max_score = 0
            for child in child_nodes:
                score = data_dict[child]['reward'] / data_dict[child]['visits']
                if score > max_score:
                    print(score)
                    max_score = score
                    max_scoring_child_node = child
            score_list.append(max_score)
            node_list.append(max_scoring_child_node)

            traverse_tree(node=max_scoring_child_node, data_dict=data_dict,
                          score_list=score_list, node_list=node_list)
        except KeyError:
            print('reached leaf node, completed iteration')
            print(node_list)
            print(score_list)


def main():

    tree_dictionary_fp = sys.argv[1]
    top_kmers_outfile = sys.argv[2]

    with open(tree_dictionary_fp) as json_file:
        data_dictionary = json.load(json_file)
    explore(data_dictionary, 50, top_kmers_outfile)


if __name__ == '__main__':
    main()