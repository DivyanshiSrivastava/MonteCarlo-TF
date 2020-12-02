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
        """
        As input, this function receives a k-mer.
        It adds all possible 2 bp flanks to this k-mer, resulting in 4 ^ 2 = 16
        new child k-mers. It returns a list of these new child k-mers.
        Parameters:
            node (str): A str parent k-mer. For example: 'TAAT'
        Returns (list): A list of 16 child k-mers.
        """
        # node = k-mer
        potential_flanks = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                            'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        child_list = []  # this is a list of k-mers
        for flanks in potential_flanks:
            child_node = flanks[0] + node + flanks[1]
            child_list.append(child_node)
        return child_list

    def check_complete_expansion_status(self, child_list):
        """
        This function looks at the state dictionary to check whether a node has
        been completely expanded. As input, this function takes a list of 16
        k-mers that are childs for the node being looked at.

        Rationale:
        The state dictionary is a defaultdict(dict). Therefore, any key that is
        assigned a value will exist.
        For example:
        Let state_dict be an empty state dict. It's behavior is as follows:
        ...
        >>> from collections import defaultdict
        >>> state_dict = defaultdict(dict)
        >>> print(state_dict)
        defaultdict(<class 'dict'>, {})
        >>> state_dict['TAAT']['reward'] = 0
        >>> print(state_dict)
        defaultdict(<class 'dict'>, {'TAAT': {'reward': 0}})
        >>> state_dict['GTAATG']
        {}
        >>> state_dict['GTAATG']['status']
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        KeyError: 'status'
        ...
        Parameters:
            child_list (list): A list of 16 child k-mers.
        Returns (bool): The status (True or False) of expansion. The status is
        True if the parent node is completely expanded, False otherwise.
        """
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
        """
        Simulates the play-out from a node.
        Current condition for end of playout is reaching 18 base pairs.
        This can be made more complex later.
        This function returns the score assigned to the final 18 bp k-mer reached
        at the end of playout.
        Parameters:
            node (str): A single k-mer from which the playout must be \
            simulated
        Returns (float): A score assigned by self.get_score to the final k-mer.
        """
        # simulate play-out from node
        # make this more complex later,
        # for now, simulate out to 18 base pairs and record score.
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

    def compute_ucb1_scores(self, child_list, parent_node):
        """
        For child i (k-mer) in child_list, this function will compute the UCB1 score
        using the following formula:

        ucb1 = reward(i)/visits(i)
               + square_root(2 * ln(visits(parents(i))) / visits(i))

        Parameters:
            child_list (str): list of child k-mers
        Returns:
            ucb1_score_list(str): list of ucb1 scores
        """
        times_parent_visited = 0
        if parent_node == self.root:
            for child in child_list:
                times_parent_visited += self.state_count_dictionary[child]['visits']
        else:
            times_parent_visited = self.state_count_dictionary[parent_node]['visits']
        ucb1_score_list = []
        # iterate over the child_list:
        for child_kmer in child_list:
            times_child_visited = self.state_count_dictionary[child_kmer]['visits']
            child_score = self.state_count_dictionary[child_kmer]['reward']
            av_child_reward = child_score / times_child_visited

            ucb1_score = av_child_reward + \
                        ((2 * np.log(times_parent_visited) / times_child_visited) ** 0.5)
            ucb1_score_list.append(ucb1_score)
        return ucb1_score_list

    def select_child_node_ucb1(self, child_list, parent_node):

        ucb1_score_list = self.compute_ucb1_scores(child_list=child_list,
                                                   parent_node=parent_node)
        if parent_node == 'TAAT':
            print("ucbscores")
            print(ucb1_score_list)
        index_at_max_score = np.argmax(ucb1_score_list)
        selected_kmer = child_list[index_at_max_score]
        return selected_kmer

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
                reward = self.simulate_from_node(child)
                try:
                    self.state_count_dictionary[child]['reward'] += 1
                    self.state_count_dictionary[child]['visits'] += 1
                except KeyError:
                    # record does not exist yey.
                    self.state_count_dictionary[child]['reward'] = 1
                    self.state_count_dictionary[child]['visits'] = 1
                # set status to visited & parent to "node"
                self.state_count_dictionary[child]['status'] = 'visited'
                self.state_count_dictionary[child]['parent'] = node
                # back-propagate reward to parents:
                self.back_propagate_reward(child, reward)
                break  # visit a single node at a time.
        return self.state_count_dictionary

    def monte_carlo(self, node):
        child_list = self.get_node_children(node)
        expansion_status = self.check_complete_expansion_status(child_list)
        # note: expansion_status will be False if parent is not \
        # completely expanded
        # record that the parent has been visited:
        try:
            self.state_count_dictionary[node]['visits'] += 1
        except KeyError:
            self.state_count_dictionary[node]['visits'] = 1
        if expansion_status is False:
            print("Node is not completely expanded")
            self.randomly_expand_parent_node(node=node,
                                             child_list=child_list)
        else:
            print("Node is completely expanded")
            child_node = self.select_child_node_ucb1(child_list, parent_node=node)
            self.monte_carlo(child_node)

    def return_state_dictionary(self):
        return self.state_count_dictionary


mc_tree = Tree(root='TAAT')

idx = 0
while idx < 164:
    idx += 1
    dictc = mc_tree.monte_carlo('TAAT')
stats = mc_tree.return_state_dictionary()
for keys in stats:
    if keys == 'TAAT':
        print(keys)
        print(stats[keys])




































