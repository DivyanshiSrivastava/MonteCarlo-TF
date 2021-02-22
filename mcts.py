"""
Implement a Monte Carlo Tree Search in k-mer space.
Use cis-bp cognate motifs to initialize the root node.
Add nucleotides on both flanks to keep cognate motifs centered.
"""
import json
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from helper import get_kmer_score_in_ns
from helper import construct_background_data
from helper import KmerScores
import sys
import subprocess


class Tree:

    def __init__(self, root, model, background_data, index_prefix, no_of_bound_seqs):
        self.root = root
        self.model = model
        self.state_count_dictionary = defaultdict(dict)
        # The root node is a k-mer from a TF's cognate motif in cis-bp.
        # Initializing the root node.
        self.state_count_dictionary[root]['reward'] = 0
        self.background_data = background_data
        self.index_prefix = index_prefix
        self.no_of_bound_seqs = no_of_bound_seqs


    @staticmethod
    def get_node_children(node):
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
        # score_at_kmer = get_kmer_score_in_ns(model=self.model, kmer=kmer)
        kmer_score_class_inst = KmerScores(kmer=kmer, model=self.model,
                                           background_list=self.background_data)
        return kmer_score_class_inst.score_sequences()

    def get_kmer_frequencies_in_seqs(self, kmer):
        # Here, we should have index_prefix k-mer indexes for kmer
        # lengths = 9, 11, 13, 15, 17, (19?).
        kmer_size = len(kmer)
        index_file_name = self.index_prefix + '.' + str(kmer_size) + '.txt'
        jf_query = subprocess.Popen(
            ['jellyfish', 'query', index_file_name, kmer],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        jf_query_stdout, stderr = jf_query.communicate()
        print(jf_query_stdout)
        # jf_query_stdout is of the following form:
        # For a valid query:
        # TAATTAATTAA 30
        # For an invalid query:
        # Invalid mer 'TAAT'
        if jf_query_stdout.strip().split()[0] is 'Invalid':
            count = 0
            print('In this weird place FYI')
        else:
            count = int(jf_query_stdout.strip().split()[1])
            print(count)
        return count / self.no_of_bound_seqs

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

        # Wednesday, January 27th: Modifying the stopping criteria.
        # Old stopping criteria: Reaching a k-mer length of 18.
        # while len(base_kmer) <= 18:
        #     # select a kmer to add:
        #     select = np.random.randint(low=0, high=15)
        #     base_kmer = potential_flanks[select][0] + base_kmer + \
        #                 potential_flanks[select][1]
        # return self.get_score(base_kmer)

        # New stopping criteria:
        # Number of Peaks: Total number of TF ChIP-seq peaks.
        # Threshold: The minimum number of times we should see the motif in \
        # the input data.
        # For example:
        # Number of ChIP-seq peaks N = 5000.
        # Threshold T = 0.01 (or 1%)
        # Stopping criteria:
        # Let M = 7 be expected length of k-mers. (Use a metric to choose M?)
        # Alternatively, use the size of the cognate motif?
        # When len(base K-mer) > M,
        # Query a pre-built index.
        # If base k-mer occurs at-least T * N times:
        # continue.
        # Else:
        # Compute Score and Return Score.
        m = 7
        while len(base_kmer) <= m:
            select = np.random.randint(low=0, high=15)
            base_kmer = potential_flanks[select][0] + base_kmer + \
                        potential_flanks[select][1]

        while self.get_kmer_frequencies_in_seqs(kmer=base_kmer) >= 0.05:
            select = np.random.randint(low=0, high=15)
            base_kmer = potential_flanks[select][0] + base_kmer + \
                        potential_flanks[select][1]

        # NUANCE NEEDED HERE?
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
                    self.state_count_dictionary[child]['reward'] += reward
                    self.state_count_dictionary[child]['visits'] += 1
                except KeyError:
                    # record does not exist yey.
                    self.state_count_dictionary[child]['reward'] = reward
                    self.state_count_dictionary[child]['visits'] = 1
                # set status to visited & parent to "node"
                self.state_count_dictionary[child]['status'] = 'visited'
                self.state_count_dictionary[child]['parent'] = node
                # back-propagate reward to parents:
                self.back_propagate_reward(child, reward)
                break  # visit a single node at a time.
        return self.state_count_dictionary

    def monte_carlo(self, node, parent_node):
        child_list = Tree.get_node_children(node)
        expansion_status = self.check_complete_expansion_status(child_list)
        # note: expansion_status will be False if parent is not \
        # completely expanded
        # record that the parent has been visited:
        try:
            self.state_count_dictionary[node]['visits'] += 1
        except KeyError:
            self.state_count_dictionary[node]['visits'] = 1

        # First, check that the parent is <=16 base pairs
        # because we are only simulating out to 18 base pairs)
        # if len(node) > 16:
        #     print("Reached play-out terminal state")
        #     # Note: The k-mer is now 18 base pairs, and is a child of a 16 bp
        #     # k-mer
        #     reward = self.get_score(node)
        #     self.state_count_dictionary[node]['parent'] = parent_node
        #     try:
        #         self.state_count_dictionary[node]['reward'] += reward
        #     except KeyError:
        #         self.state_count_dictionary[node]['reward'] = reward

        # END of playout is reached when we reach a kmer with frequency less
        # than 5%.
        if len(node) > 7:
            if self.get_kmer_frequencies_in_seqs(node) < 0.05:
                print("Reached play-out terminal state")
                reward = self.get_score(node)
                self.state_count_dictionary[node]['parent'] = parent_node
                try:
                    self.state_count_dictionary[node]['reward'] += reward
                except KeyError:
                    self.state_count_dictionary[node]['reward'] = reward
            else:
                pass

        else:
            if expansion_status is False:
                # print("Node is not completely expanded")
                self.randomly_expand_parent_node(node=node,
                                                 child_list=child_list)
            else:
                # print("Node is completely expanded")
                child_node = self.select_child_node_ucb1(child_list, parent_node=node)
                self.monte_carlo(child_node, parent_node=node)

    def return_state_dictionary(self):
        return dict(self.state_count_dictionary)


def construct_kmer_indexes(bound_tf_fa_file, index_prefix):
    # use jellyfish to define the 9-mer, 11-mer, 13-mer, 15-mer \
    # and 17-mer indexes
    for K in [9, 11, 13, 15, 17]:
        str_name = index_prefix + '.' + str(K) + '.txt'
        print(str_name)
        subprocess.Popen(['jellyfish', 'count', '-m',
                          str(K), '-s', '1M', '-t', '10',
                          '-C', bound_tf_fa_file,
                          '-o', index_prefix + '.' + str(K) + '.txt'])


def run_mcts(num_of_iterations, root_kmer, outfile_prefix):
    idx = 0
    while idx < num_of_iterations:
        idx += 1
        mc_tree.monte_carlo(root_kmer, parent_node=None)

    iter_result = mc_tree.return_state_dictionary()
    outfile = outfile_prefix + str(num_of_iterations) + '.json'
    with open(outfile, 'w') as fp:
        json.dump(iter_result, fp)


if __name__ == "__main__":

    model_path = sys.argv[1]
    root_kmer = sys.argv[2]
    out = sys.argv[3]
    input_bound_seqs_file = sys.argv[4]

    model = load_model(model_path)
    background_data = construct_background_data(size=10)

    input_bound_seq_fa = np.loadtxt(input_bound_seqs_file, dtype=str)
    no_of_input_seqs = len(input_bound_seq_fa) / 2

    kmer_indexfiles_prefix = out + 'kmerindex'

    # construct K-mer indexes:
    construct_kmer_indexes(bound_tf_fa_file=input_bound_seqs_file,
                           index_prefix=kmer_indexfiles_prefix)

    mc_tree = Tree(root=root_kmer, model=model,
                   background_data=background_data, index_prefix=kmer_indexfiles_prefix,
                   no_of_bound_seqs=no_of_input_seqs)

    for num_of_iters in [1000, 5000, 10000, 50000, 100000]:
        run_mcts(num_of_iterations=num_of_iters, root_kmer=root_kmer,
                 outfile_prefix=out)
