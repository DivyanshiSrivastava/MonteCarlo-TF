"""
Implement a Monte Carlo Tree Search in k-mer space.
Use cis-bp cognate motifs to initialize the root node.
Add nucleotides on both flanks to keep cognate motifs centered.
"""

import numpy as np
from collections import defaultdict


class Tree:

    def __init__(self, root):
        self.root = root
        self.state_count_dictionary = defaultdict(dict)
        # The root node is a k-mer from a TF's cognate motif in cis-bp.
        # Initializing the root node.
        self.state_count_dictionary[root]['reward'] = 0

    def get_node_children(self, node):
        # node = k-mer
        potential_flanks = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                            'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        child_list = []  # this is a list of k-mers
        for flanks in potential_flanks:
            child_node = flanks[0] + node + flanks[1]
            child_list.append(child_node)
        return child_list

    def check_complete_expansion_status(self, child_list):
        default_status = True  # assume: node is completely expanded.
        status = default_status

        # child_list should be a list of 16 k-mers.
        # check if each child has been visited at-least once
        # note: not incrementing anything.
        for child in child_list:
            if self.state_count_dictionary[child] == {} or \
                    self.state_count_dictionary[child]['status'] == 'unvisited':
                # i.e. child has never been visited, no record exists.
                # i.e. parent node is not completely expanded.
                status = False
                # additionally, set child status to unvisited:
                self.state_count_dictionary[child]['status'] = 'unvisited'
            else:
                pass
        return status

    def get_score(self, kmer):
        return 1

    def simulate_from_node(self, node):
        # simulate play-out from node
        # make this more complex later,
        # for now, simulate out to 20 base pairs and record score.
        base_kmer = node
        potential_flanks = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                            'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        while len(base_kmer) <= 18:
            # select a kmer to add:
            select = np.random.randint(low=0, high=15)
            base_kmer = potential_flanks[select][0] + base_kmer + \
                        potential_flanks[select][1]
        return self.get_score(base_kmer)

    def back_propagate_reward(self, node, reward):
        if node == self.root:
            return None
        else:
            parent = self.state_count_dictionary[node]['parent']
            self.state_count_dictionary[parent]['reward'] += reward
            self.back_propagate_reward(parent, reward)

    def randomly_expand_parent_node(self, node, child_list):
        # child_list is a list of 16 k-mers
        # if we're in this function, we know that the parent node "node" is \
        # not completely expanded.

        # if parent-node is only partially expanded, choose a child node \
        # to visit randomly from amongst child-nodes that are unvisited.
        for child in child_list:
            # note: if I am inside this function, \
            # fn: "check_complete_expansion_status" has already been run for \
            # parent node "node".
            # i.e. each unvisited child node has an associated unvisited tag.
            if self.state_count_dictionary[child]['status'] == 'unvisited':
                print(self.state_count_dictionary)
                print(self.state_count_dictionary[child])
                reward = self.simulate_from_node(child)
                try:
                    self.state_count_dictionary[child]['reward'] += 1
                except KeyError:
                    # record does not exist yey.
                    self.state_count_dictionary[child]['reward'] = 1
                # set status to visited & parent to "node"
                self.state_count_dictionary[child]['status'] = 'visited'
                self.state_count_dictionary[child]['parent'] = node
                # back-propagate reward to parents:
                self.back_propagate_reward(child, reward)
                break  # visit a single node at a time.
        return self.state_count_dictionary

    def monte_carlo(self):
        root_child_list = self.get_node_children(self.root)
        expansion_status = self.check_complete_expansion_status(root_child_list)
        # note: expansion_status will be False if parent is not \
        # completely expanded
        if expansion_status is False:
            print("Root is not completely expanded")
        self.randomly_expand_parent_node(node=self.root,\
                                                      child_list=root_child_list)

    def return_state_dict(self):
        return self.state_count_dictionary


mc_tree = Tree(root='TAAT')

idx = 0
while idx < 100:
    idx += 1
    mc_tree.monte_carlo()

dictc = mc_tree.return_state_dict()

print("HERE")
for key in dictc:
    print(key)
    print(dictc[key])




































