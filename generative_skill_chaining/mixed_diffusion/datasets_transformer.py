import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import pickle

class StandardDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MixedDiffusionDataset(Dataset):
    #####################################################
    # This class is used to load the dataset for the    #
    # diffusion transformer scorenet and classifier     #
    #####################################################

    def __init__(self, dataset_path, obs_processor):
        #####################################################
        # dataset_path: path to the dataset                 #
        # obs_processor: function to process the observation#
        #####################################################

        self.dataset_path = dataset_path
        self.obs_processor = obs_processor

        self.eval_datapath = self.dataset_path.split(".")[0] + "_eval_random.pkl" # for the classifier
        self.neg_eval_datapath = self.dataset_path.split(".")[0] + "_eval_random_neg.pkl" # for the classifier

        with open(self.eval_datapath, 'rb') as f:
            self.data = pickle.load(f)
            self.data = np.random.permutation(self.data).tolist()
            self.data = self.data[:10000]


        with open(dataset_path, 'rb') as f:
            self.data.extend(pickle.load(f))

        with open(self.neg_eval_datapath, 'rb') as f:
            self.neg_data = pickle.load(f)
            self.neg_data = np.random.permutation(self.neg_data)[:10000]

        self.obs1 = []
        self.obs2 = []
        self.act = []
        self.obs_indices = []

        self.neg_obs1 = []
        self.neg_obs = []

        for data in tqdm.tqdm(self.data):

            if len(data['observations']) != 2:
                continue

            observation1 = data['observations'][0]
            observation2 = data['observations'][1]
            action = data['actions'][0]
            reset_info = data['reset_info'][0] if type(data['reset_info']) == list else data['reset_info']

            self.obs1.append(
                self.obs_processor(observation1, reset_info['policy_args'])
            )

            self.obs2.append(
                self.obs_processor(observation2, reset_info['policy_args'])
            )

            self.obs_indices.append(reset_info['policy_args']['observation_indices'])

            self.act.append(action)

            if len(self.obs1) > 10000:
                break

        self.obs1 = np.array(self.obs1)*2 # scale to [-1, 1] from [-1/2, 1/2]
        self.obs2 = np.array(self.obs2)*2 # scale to [-1, 1] from [-1/2, 1/2]
        self.act = np.array(self.act)
        self.obs_indices = np.array(self.obs_indices)


        self.neg_obs_indices = []

        for data in tqdm.tqdm(self.neg_data):
            observation1 = data['observations'][0]
            observation = data['observations'][1]
            reset_info = data['reset_info'][0] if type(data['reset_info']) == list else data['reset_info']

            self.neg_obs.append(
                self.obs_processor(observation, reset_info['policy_args'])
            )

            self.neg_obs_indices.append(reset_info['policy_args']['observation_indices'])

        self.neg_obs = np.array(self.neg_obs)*2 # scale to [-1, 1] from [-1/2, 1/2]
        self.neg_obs_indices = np.array(self.neg_obs_indices)

    def get_data_for_mode(self, mode):
        #####################################################
        # mode: "transition", "state", "classifier"         #
        #####################################################

        if mode == "transition":
            data = np.concatenate([self.obs1, self.act, self.obs2, self.obs_indices], axis=1)
        elif mode == "state":
            indices = np.random.permutation(len(self.obs1))
            data = self.obs1[indices]
            obs_indices = self.obs_indices[indices]
            data = np.concatenate([data, obs_indices], axis=1)
        elif mode == "classifier":
            pos_data = np.concatenate([self.obs2, self.obs_indices, np.ones([len(self.obs2), 1])], axis=1)[np.random.permutation(len(self.obs2))][:8000]
            neg_data = np.concatenate([self.neg_obs, self.neg_obs_indices, np.zeros([len(self.neg_obs), 1])], axis=1)[np.random.permutation(len(self.neg_obs))][:8000]
            data = np.concatenate([pos_data, neg_data], axis=0)

        return data

def get_primitive_loader(dataset_path, obs_processor, modes, batch_size=128):
    #####################################################
    # dataset_path: path to the dataset                 #
    # obs_processor: function to process the observation#
    # modes: list of modes                              #
    # batch_size: batch size                            #
    # returns: list of (dataloader, dataset) for each   #
    #          mode                                     #
    #####################################################
    
    dataset = MixedDiffusionDataset(dataset_path, obs_processor)

    all_dataset_loader = []

    for mode in modes:
        data = dataset.get_data_for_mode(mode)
        mode_dataset = StandardDataset(data)
        mode_loader = DataLoader(mode_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        all_dataset_loader.append((mode_loader, mode_dataset))
    
    return all_dataset_loader