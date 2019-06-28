import numpy as np 
import os

from exc1 import PathHMM, viterbi, compute_observation_probabilities, evaluate_hmms

import ipdb

def compute_means_and_variances(frame_features_list, decodings):
    assert(len(frame_features_list) == len(initial_decodings))
    bins = {}
    # Watch out, magic number alert!
    # 152... number of states
    # 64... dimension of framewise feature vectors
    means = np.zeros((152, 64))
    variances = np.zeros((152, 64))
    for i in range(len(frame_features_list)):
        decoding = decodings[i]
        features = frame_features_list[i]
        for j in range(len(decoding)):
            if not int(decoding[j]) in bins:
                bins[int(decoding[j])] = []
            bins[int(decoding[j])].append(features[:,j])

    for state in bins:
        observation = np.array(bins[state])
        mean = np.mean(observation, axis=0)
        var = np.var(observation, axis=0)
        means[state,:] = mean
        variances[state,:] = var
    return means, variances


if __name__ == "__main__":
    NUM_EPOCHS = 5

    with open('./exc1/hmm_definition.dict', 'r') as file:
        all_hmm_strings = file.read().splitlines()

    trained_hmms = []

    for hmm_string in all_hmm_strings:
        hmm = PathHMM([hmm_string])
        # ipdb.set_trace()
        all_training_files = [f for f in os.listdir('./exc2') if hmm_string in f]

        training_files = [f for f in all_training_files if 'initStates' not in f]
        training_files.sort(key=lambda s: int(s.split('.')[0].split('_')[-1]))

        init_files = [f for f in all_training_files if 'initStates' in f]
        init_files.sort(key=lambda s: int(s.split('_')[-2]))

        frame_features_list = [np.load('./exc2/' + training_file).T for training_file in training_files]
        initial_decodings = [np.load('./exc2/' + f).astype(np.int)[0].tolist() for f in init_files]

        # init hmm from init files or just manually uniformly
        hmm.update_transition_probabilities(initial_decodings)
        means, variances = compute_means_and_variances(frame_features_list, initial_decodings)

        for i in range(1, NUM_EPOCHS + 1):
            print('Starting to train "{}", iteration: {}'.format(hmm_string, i))
            decodings = []

            for frame_features in frame_features_list:
                
                observation_probabilities = compute_observation_probabilities(frame_features,
                                                                              hmm.all_possible_states_strings,
                                                                              means,
                                                                              variances)
                most_likely_path, _ = viterbi(frame_features, hmm, observation_probabilities)
                decodings.append(most_likely_path)

            # if hmm.hmm_strings == ['take_bowl']:
                # ipdb.set_trace()
            hmm.update_transition_probabilities(decodings)
            means, variances = compute_means_and_variances(frame_features_list, decodings)

            # print(hmm.transition_probabilities, hmm.transition_probabilities.shape)

        trained_hmms.append(hmm)
    
    ipdb.set_trace()

    hmm_paths = ['./exc1/test1.grammar', './exc1/test2.grammar', './exc1/test3.grammar']
    video_paths = ['./exc1/P03_cam01_P03_cereals.npy', './exc1/P03_cam01_P03_coffee.npy', './exc1/P03_cam01_P03_milk.npy']
    gt_paths = ['./exc1/P03_cam01_P03_cereals.gt', './exc1/P03_cam01_P03_coffee.gt', './exc1/P03_cam01_P03_milk.gt']
    observation_probabilities_paths = {'./exc1/P03_cam01_P03_cereals.npy':'./obsprobs/obs_probs_cereals.npy',
                                       './exc1/P03_cam01_P03_coffee.npy':'./obsprobs/obs_probs_coffee.npy',
                                       './exc1/P03_cam01_P03_milk.npy':'./obsprobs/obs_probs_milk.npy'}
    means = np.loadtxt('./exc1/GMM_mean.matrix')
    variances = np.loadtxt('./exc1/GMM_var.matrix')

    hmms = []
    for hmm_path in hmm_paths:
        with open(hmm_path, 'r') as file:
            hmm_strings = file.read().splitlines()[0].split(' ')
        path_hmm = PathHMM(hmm_strings)
        hmms.append(path_hmm)
    
    evaluate_hmms(hmms, video_paths, gt_paths, observation_probabilities_paths, means, variances)