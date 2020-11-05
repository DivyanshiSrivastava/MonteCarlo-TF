"""
Define a Monte Carlo Search Tree that finds k-mers driving NN predictions.
Even if this isn't required, it's a good programming exercise.
"""

import time
import numpy as np
import math
import sys
from keras.models import load_model
from collections import defaultdict

np.random.seed(1)

# Pseudo Code
# a. Implement a Monte Carlo Tree Search over the k-mer space.
# b. Let it go out to 20 base pairs.
# Steps:
# 0. Do completely random simulations to determine initial starting values for a set of nodes.
# 1. Selection of path using the UCB1 algorithm.
# 2. Expansion:
#    a. When UCB1 can no longer be applied at any node ( i.e. If no child nodes have been expanded,
#       an unvisited node is chosen at random & a new record is added to the tree. eg. 0/0)
# 2. Randomly simulate from this child-node using random MC simulation or weighted simulation & update node.
# 3. Select a new node using a metric of exploitation vs. expansion
# 4. Repeat from the top?
# END


class MonteCarlo(object):

    def __init__(self, model):
        self.model = model
        self.curr_state = 'T'

    def check_node_status(self, curr_kmer, state_dict):
        # Keep track of whether this node has been expanded
        # And also if which child nodes have been visited
        explored_status = True
        unexplored_list = []
        for nucleotide in ['rA', 'rT', 'rG', 'rC', 'lA', 'lT', 'lG', 'lC']:
            letter = nucleotide[1]  # Extract the nucleotide information
            direction = nucleotide[0]
            # Construct the child nodes
            if direction == 'r':
                curr_child_node = curr_kmer + letter
            else:
                curr_child_node = letter + curr_kmer
            # Check status:
            if curr_child_node in state_dict.iterkeys():
                pass
            else:
                explored_status = False
                unexplored_list.append(nucleotide)
        return explored_status, unexplored_list

    def select_node(self, curr_kmer, state_dict):
        # use UCB1 or something similar to choose appropriate child node
        # This function assumes that the parent node has been expanded.
        # Using UCB1 here, to construct a confidence bound on each node:
        # Have to check if it's appropriate given that xi lies between 0 and 1.

        print("The current k-mer is:")
        print(curr_kmer)

        # Find the total number of times this particular choice has been visited out of all possible options.
        n = 0
        nucleotide_list = ['rA', 'rT', 'rG', 'rC', 'lA', 'lT', 'lG', 'lC']
        for nucleotide in nucleotide_list:
            direction = nucleotide[0]
            letter = nucleotide[1]  # Removing the directionality
            if direction == 'r':
                potential_kmer = curr_kmer + letter
            else:
                potential_kmer = letter + curr_kmer
            n = n + len(state_dict[potential_kmer])
        print("The potential-kmer is:")
        print(potential_kmer)
        print("The total number of times this parent has been visited:")
        print(len(state_dict[potential_kmer]))

        # Find the potential k-mer with the highest UCB value and return:
        max = -1
        average_scores = []
        selected_kmer = curr_kmer
        for nucleotide in nucleotide_list:
            direction = nucleotide[0]
            letter = nucleotide[1]  # Removing the directionality
            if direction == 'r':
                potential_kmer = curr_kmer + letter
            else:
                potential_kmer = letter + curr_kmer
            mean = np.mean(state_dict[potential_kmer])
            ni = len(state_dict[potential_kmer])
            average_scores.append(mean)
            # cb = mean + (2 * math.log(n)/ni) ** 0.5
            cb = mean
            if cb > max:
                max = cb
                selected_kmer = potential_kmer
        # Do this probabilistically:
        # Generate a random uniform number:
        px = np.random.uniform(0, 1)
        if px > 0.50:
            pass
            print("choosing the path with the highest mean")
        else:
            print("choosing a random path in the tree")
            # This event is relatively rare
            idx = np.random.choice(8)
            letter = nucleotide_list[idx][1]
            direction = nucleotide_list[idx][0]
            if direction == 'r':
                selected_kmer = curr_kmer + letter
            else:
                selected_kmer = letter + curr_kmer
        return selected_kmer

    def back_propagate_scores(self, selected_kmer, visited_kmers, state_dict):
        modelscores = Scores(selected_kmer, self.model)
        print(selected_kmer)
        score = modelscores.score_sequences()
        for kmer in visited_kmers:
            state_dict[kmer].append(score)
        return state_dict

    def simulate(self, state_dict):
        # Define the integer to nucleotide mappings
        # Use integers to randomly choose paths
        visited_kmers = []
        curr_kmer = self.curr_state

        selected_kmer = curr_kmer
        int_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}

        while len(selected_kmer) < 10:
            # explored, unexplored_list = self.check_node_status(curr_kmer, state_dict)
            # if explored:
            #     # All child nodes have been explored, choose node based on UCB
            #     kmer = self.select_node(curr_kmer, state_dict)
            #     visited_kmers.append(kmer)
            #    curr_kmer = kmer
            #if True:
            unexplored_list = ['A', 'T', 'G', 'C']  # 'lA', 'lT', 'lG', 'lC']
            choice = np.random.choice(len(unexplored_list))
            nucleotide = unexplored_list[choice]
            # Not using direction for now
            selected_kmer = selected_kmer + nucleotide
            print(selected_kmer)
            visited_kmers.append(selected_kmer)
            modelscores = Scores(selected_kmer, self.model)
            score = modelscores.score_sequences()
        for kmer in visited_kmers:
            state_dict[kmer].append(score)

    def run_simulation(self):
        state_dict = defaultdict(list)
        # Play out a random game from the current position
        begin_time = time.time()
        max_time = 60
        while time.time() - begin_time < max_time:
            self.simulate(state_dict)
        l = []
        for key, val in state_dict.iteritems():
            l.append([key, np.median(val)])
        l = np.asarray(l)
        print(np.asarray(l[np.argsort(l[:, 1].astype(float))]))
        # print l[:, 0]
        # print state_dict
        print(len(state_dict))



class Scores(object):

    def __init__(self, kmer, model):
        self.kmer = kmer
        self.size_of_background = 100  # declared here.
        self.seq_length = 500
        self.model = model

    def background_data(self):
        # letters for simulation
        letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        background_list = []
        for idx in range(self.size_of_background):
            sequence = np.random.randint(0, 4, 500)
            sequence = [letter[x] for x in sequence]
            background_list.append(sequence)
        return background_list

    def make_onehot(self, buf):
        # Nucleotide dictionary here:-
        fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        onehot = [fd[base] for seq in buf for base in seq]
        onehot_np = np.reshape(onehot, (-1, self.seq_length, 4))
        return onehot_np

    def embed(self, curr_background):
        # Note: using a fixed insert position here.
        pos = 250
        sequence_with_kmer = curr_background[:]
        sequence_with_kmer[pos: pos + len(self.kmer)] = self.kmer
        return sequence_with_kmer

    def simulate_data(self):
        background_list = self.background_data()
        seqlist = []
        # Embed the motif of choice:
        for curr_background in background_list:
            sequence_with_kmer = self.embed(curr_background)
            seqlist.append(''.join(sequence_with_kmer))  # Doing the join after the embedding the kmer
        # extracting the sequence data
        dat = np.array(seqlist)
        dat = self.make_onehot(dat)
        return dat

    def score_sequences(self):
        # generate the simulated data
        dat = self.simulate_data()
        # score the simulated data
        return np.median(self.model.predict(dat))


if __name__ == "__main__":
    modelpath = sys.argv[1]
    model = load_model(modelpath)
    mc = MonteCarlo(model)
    mc.run_simulation()
