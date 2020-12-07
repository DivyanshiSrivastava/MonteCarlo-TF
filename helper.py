"""
Functions for scoring an input k-mer.
"""

import numpy as np


def get_onehot(sequence_list, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
          'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in sequence_list for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def get_kmer_score_in_ns(model, kmer):
    """
    Parameters:
        kmer (str): A sequence string to be scored by the network
        model (Keras model): A Keras model to be used for scoring.
    Note: seq_length is hardcoded to 500 as my models are trained with
    500 base pair sequences. Change this if/when making a package. TODO.
    """
    simulated_sequence = np.repeat('N', 500)
    kmer_length = len(kmer)
    seq_length = 500

    embedding_start_idx = 240
    embedding_end_idx = 240 + kmer_length

    simulated_sequence[embedding_start_idx: embedding_end_idx] = kmer
    simulated_sequence_list = list(simulated_sequence)
    simulated_sequence_onehot = get_onehot(simulated_sequence_list,
                                           seq_length=seq_length)
    model_score = float(model.predict(simulated_sequence_onehot)[0])
    return model_score


def construct_background_data(size):
    letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    background_list = []
    for idx in range(size):
        sequence = np.random.randint(0, 4, 500)
        sequence = [letter[x] for x in sequence]
        background_list.append(sequence)
    return background_list


class KmerScores(object):
    def __init__(self, kmer, model, background_list):
        self.kmer = kmer
        self.seq_length = 500
        self.model = model
        self.background_list = background_list

    def get_onehot(self, buf):
        fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
              'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
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
        seqlist = []
        # Embed the motif of choice:
        for curr_background in self.background_list:
            sequence_with_kmer = self.embed(curr_background)
            seqlist.append(''.join(sequence_with_kmer))  # Doing the join after the embedding the kmer
        # extracting the sequence data
        dat = np.array(seqlist)
        dat = self.get_onehot(dat)
        return dat

    def score_sequences(self):
        # generate the simulated data
        dat = self.simulate_data()
        # score the simulated data
        median_score = np.median(self.model.predict(dat))
        formatted_med_score = float("{0:.2f}".format(median_score))
        return formatted_med_score


