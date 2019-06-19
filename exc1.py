import numpy as np
import math
from scipy.stats import multivariate_normal
import os
import ipdb

    
class PathHMM():
    def __init__(self, hmm_strings,
                 states_per_hmm_path = './exc1/hmm_definition.vector',
                 all_possible_states_path = './exc1/hmm_state_definition.vector'):
        assert(type(hmm_strings) == list)
        with open(states_per_hmm_path, 'r') as file:
            self.num_states_per_hmm = file.read().splitlines()
        with open(all_possible_states_path, 'r') as file:
            self.all_possible_states_strings = file.read().splitlines()
        self.num_all_possible_states = len(self.all_possible_states_strings)
        self.hmm_strings = hmm_strings

        self.state_strings = []
        for hmm_string in hmm_strings:
            for possible_state in self.all_possible_states_strings:
                if hmm_string in possible_state:
                    self.state_strings.append(possible_state)

        self.priors = np.zeros(len(self.all_possible_states_strings))
        first_state_index = self.all_possible_states_strings.index(self.state_strings[0])
        self.priors[first_state_index] = 1  # First HMM deterministically starts in its first state

        self.transition_probabilities_list = []
        self.transition_probabilities = np.zeros((self.num_all_possible_states, self.num_all_possible_states))
        self.__init_transition_probabilities()
    
    def __init_transition_probabilities(self):
        for state_index, state in enumerate(self.state_strings):
            global_state_index = self.all_possible_states_strings.index(state)
            self.transition_probabilities[global_state_index, global_state_index] = 0.9
            if state_index + 1 < len(self.state_strings):
                next_state = self.state_strings[(state_index+1)]
                next_global_state_index = self.all_possible_states_strings.index(next_state)
                self.transition_probabilities[next_global_state_index, global_state_index] = 0.1
    
    def update_transition_probabilities(self, decodings):
        transitions = np.zeros((self.num_all_possible_states, self.num_all_possible_states))
        occurences = np.zeros(self.num_all_possible_states, dtype=np.int)
        for decoding in decodings:
            for i, state in enumerate(decoding):
                occurences[state] += 1
                if i < len(decoding) - 1:
                    next_state = decoding[i + 1]
                else:
                    #next_state = ??  # TODO: find out how to handle transitions after last state
                transitions[next_state,state] += 1
        occurences[occurences == 0] = 1  # to circumvent div-by-zero. value shouldn't matter, watch out for end-state value!!!
        trans_probs = transitions / occurences[None,:]
        # self.transition_probabilities_list.append(trans_probs)
        self.transition_probabilities = trans_probs
        self.terminal_state_transition = len(decodings) / occurences[occurences > 0][-1]
    

def viterbi(observed_sequence, path_hmm, observation_probabilities):
    # observed_sequence is sequence of frame-wise video features with shape (64, NUM_VIDEO_FRAMES)
    max_sequence = []  # contains indices of global states, i.e. in range(153)

    deltas = -np.log(path_hmm.priors) + observation_probabilities[0]
    argmins = []
    # previous_most_likely_state_index = np.argmin(deltas)
    # max_sequence.append(previous_most_likely_state_index)
    # likeliest_path_probability = deltas[previous_most_likely_state_index]

    # print(deltas, np.argmin(deltas))
    # ipdb.set_trace()

    num_video_frames = observed_sequence.shape[1]
    for i in range(1, num_video_frames):

        deltas = deltas - np.log(path_hmm.transition_probabilities)
        argmins.append(np.argmin(deltas, axis=1))
        deltas = np.min(deltas, axis=1)
        deltas = deltas + observation_probabilities[i]

        # print(deltas, np.argmin(deltas))
        # ipdb.set_trace()

        # previous_most_likely_state_index = np.argmin(deltas)
        # likeliest_path_probability = deltas[previous_most_likely_state_index]
        # max_sequence.append(previous_most_likely_state_index)
    
    likeliest_last_state, likeliest_path_probability = np.argmin(deltas), np.min(deltas)
    max_sequence.append(likeliest_last_state)
    argmins.reverse()
    for argmin in argmins:
        likeliest_last_state = argmin[likeliest_last_state]
        max_sequence.append(likeliest_last_state)
    max_sequence.reverse()
    return max_sequence, likeliest_path_probability


def compute_observation_probabilities(frame_features, all_possible_states_strings, means, variances):
    input_dim, num_video_frames = frame_features.shape
    num_possible_states = len(all_possible_states_strings)
    observation_probabilities = np.zeros((num_video_frames, num_possible_states))
    for i in range(num_video_frames):
        for j in range(num_possible_states):
            cov = np.zeros((input_dim, input_dim))
            for k in range(input_dim):
                cov[k,k] = variances[j,k]
            if not cov.any():
                observation_probabilities[i, j] = 0
            else:
                observation_probabilities[i,j] = -multivariate_normal.logpdf([frame_features[:,i]], means[j], cov=cov)
    return observation_probabilities


def get_observation_probabilities(obsprobs_path, frame_features, all_possible_states_strings, means, variances):
    if os.path.isfile(obsprobs_path):
        print('Loading observation probabilities {}'.format(obsprobs_path))
        observation_probabilities = np.load(obsprobs_path)
    else:
        print('Computing observation probabilities {}'.format(obsprobs_path))
        observation_probabilities = compute_observation_probabilities(frame_features, all_possible_states_strings, means, variances)
        np.save(obsprobs_path, observation_probabilities)
    return observation_probabilities


def my_own_multivariate_gaussian(x, mean, cov):
    pass


def mean_over_frames(gt, max_sequence_states):
    assert(len(gt) == len(max_sequence_states))
    corrects = 0
    for i, state in enumerate(max_sequence_states):
        if gt[i] in state:
            corrects += 1
    return corrects / len(max_sequence_states)


if __name__ == "__main__":
    hmm_paths = ['./exc1/test1.grammar', './exc1/test2.grammar', './exc1/test3.grammar']
    video_paths = ['./exc1/P03_cam01_P03_cereals.npy', './exc1/P03_cam01_P03_coffee.npy', './exc1/P03_cam01_P03_milk.npy']
    gt_paths = ['./exc1/P03_cam01_P03_cereals.gt', './exc1/P03_cam01_P03_coffee.gt', './exc1/P03_cam01_P03_milk.gt']
    observation_probabilities_paths = {video_paths[0]:'./obsprobs/obs_probs_cereals.npy', video_paths[1]: './obsprobs/obs_probs_coffee.npy', video_paths[2]: './obsprobs/obs_probs_milk.npy'}

    means = np.loadtxt('./exc1/GMM_mean.matrix')
    variances = np.loadtxt('./exc1/GMM_var.matrix')
    ipdb.set_trace()

    for hmm_path in hmm_paths:

        with open(hmm_path, 'r') as file:
            hmm_strings = file.read().splitlines()[0].split(' ')

        path_hmm = PathHMM(hmm_strings)

        for i, video_path in enumerate(video_paths):

            frame_features = np.load(video_path)
            observation_probabilities = get_observation_probabilities(observation_probabilities_paths[video_path],
                                                                      frame_features,
                                                                      path_hmm.all_possible_states_strings,
                                                                      means,
                                                                      variances)
            ipdb.set_trace()

            max_sequence_indices, likeliest_path_probability = viterbi(frame_features, path_hmm, observation_probabilities)

            gt_path = gt_paths[i]
            with open(gt_path, 'r') as file:
                gt = file.read().splitlines()
            
            max_sequence_states = [path_hmm.all_possible_states_strings[index] for index in max_sequence_indices]
            acc = mean_over_frames(gt, max_sequence_states)
    
            print('Processing video:', video_path)
            print('Using HMM path:', hmm_path)
            print('Contained states:', hmm_strings)
            # print()
            print('Probability:' ,likeliest_path_probability)
            print('Accuracy:', acc)
            print('---')