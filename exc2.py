import numpy as np 
import os

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
                state_without_transition = 
            else:
                states_till_transition = change_index - change_indexes[i-1]
            self.transition_probabilities[i,i] = 
            self.transition_probabilities[i+i,i] = 
        ipdb.set_trace()



if __name__ == "__main__":
    with open('./exc1/hmm_definition.dict', 'r') as file:
        all_hmm_strings = file.read().splitlines()
    
    with open('./exc1/hmm_state_definition.vector', 'r') as file:
        all_possible_states_strings = file.read().splitlines()
    
    training_files = os.listdir('./exc2')

    for hmm_string in all_hmm_strings:
        hmm_training_files = [f for f in training_files if (hmm_string in f and not 'initStates' in f)]

        for training_file in hmm_training_files:
            init_file = training_file.split('.')[0] + '_initStates.npy'
            init_states = np.load('./exc2/' + init_file)
            hmm = HMM(hmm_string, init_states)

            framewise_featues = np.load('./exc2/' + training_file)
            print(framewise_featues.shape)

