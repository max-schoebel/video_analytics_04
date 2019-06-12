
import numpy as np 
import os

from exc1 import PathHMM, viterbi, get_observation_probabilities

import ipdb


class HMM():
    def __init__(self, action_class_string, init_states):
        self.action_class_string = action_class_string
        self.num_states = int(np.max(init_states)) - int(np.min(init_states)) + 1
        self.init_states = init_states[0]
        self.transition_probabilities = np.zeros((self.num_states, self.num_states))
        self.__init_transition_probabilities()

    def __init_transition_probabilities(self):
        # for i in range(self.num_states):
            # self.transition_probabilities[i,i] = 0.9
            # if i < self.num_states:
                # self.transition_probabilities[i+1,i] = 0.1

        # array([0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.])      
        # array([4, 9, 14, 19, 24])

        bins = {}
        for state in self.init_states:
            if state in bins.keys():
                bins[state] += 1
            else:
                bins[state] = 1
            


        state_diffs = [self.init_states[i+1] - self.init_states[i] for i in range(len(self.init_states) -1)]
        change_indexes = np.where(state_diffs == 1)
        for i, change_index in enumerate(change_indexes):
            if i == 0:
                states_till_transition = change_index + 1
            elif i == len(change_indexes) - 2:
                state_without_transition = 2
            else:
                states_till_transition = change_index - change_indexes[i-1]
            self.transition_probabilities[i,i] = 2
            self.transition_probabilities[i+i,i] = 2
        ipdb.set_trace()



if __name__ == "__main__":
    NUM_EPOCHS = 5

    with open('./exc1/hmm_definition.dict', 'r') as file:
        all_hmm_strings = file.read().splitlines()

    means = np.loadtxt('./exc1/GMM_mean.matrix')
    variances = np.loadtxt('./exc1/GMM_var.matrix')

    for hmm_string in all_hmm_strings:
        hmm = PathHMM(hmm_string)
        all_training_files = [f for f in os.listdir('./exc2') if hmm_string in f]

        training_files = [f for f in all_training_files if 'initStates' not in f]
        init_files = [f for f in all_training_files if 'initStates' in f]

        initial_decodings = [np.load('./exc2/' + f).astype(np.int) for f in init_files]

        # init hmm from init files or just manually uniformly
        # initial_decodings = ... (init_files)
        hmm.update_transition_probabilities(initial_decodings)

        for i in range(1, NUM_EPOCHS + 1):
            print('Starting to train {}, iteration: '.format(hmm_string, i))
            decodings = []
            for training_file in training_files:
                frame_features = np.load('./exc2/' + training_file)
                frame_features = frame_features.T
                
                observation_probabilities_path = './obsprobs/' + training_file.split('.')[0] + '_obsprobs.npy'
                observation_probabilities = get_observation_probabilities(observation_probabilities_path,
                                                                          frame_features,
                                                                          hmm.all_possible_states_strings,
                                                                          means,
                                                                          variances)
                most_likely_path = viterbi(frame_features, hmm, observation_probabilities)
                decodings.append(most_likely_path)

            hmm.update_transition_probabilities(decodings)
            print(hmm.transition_probabilities, hmm.transition_probabilities.shape)
            # compute diff in transition probabilities
            # print diff in transition_probabilities
        ipdb.set_trace()