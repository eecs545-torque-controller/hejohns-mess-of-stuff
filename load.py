#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas
import numpy as np
import glob
import os
import re
# our files
import config

# our pytorch data loader
# these three methods are required
class LSTMDataset(Dataset):
    def __init__(self, subjects, activities):
        #############################
        # initialize parameters
        self.window_size = config.window_size
        #############################
        # get subjects and activities
        self.subjects = subjects
        self.activities = {}
        for s in self.subjects:
            self.activities[s] = [f for f in os.listdir(s) if activities.search(f)]
        self.preprocessed_data = {}
        for s in self.subjects:
            self.preprocessed_data[s] = {}
            for a in self.activities[s]:
                path = os.path.join(".", s, a, "preprocessed_data.csv")
                assert os.path.isfile(path)
                df = pandas.read_csv(path, index_col="time")
                # if we're counting number of usable windows, it's +1
                num_windows = len(df.index) - self.window_size + 1
                # store the number of usable windows so we don't have to
                # constantly read_csv to recalculate it
                self.preprocessed_data[s][a] = (path, num_windows)
        #############################
        self.len = self.calc_len()
        print(f"LSTMDataset has {self.len} windows")
        return
        #############################
        self.dfs = []
        for d in datas:
            self.dfs.append(pandas.read_csv(d, index_col="time"))
        print(self.dfs)
        self.seq_len = 100
        window_size = self.window_size
        X, y = [], []
        self.sensor_list = [
            'knee_angle_l_moment',
            'ankle_angle_l_moment',
            'subtalar_angle_l_moment',
            'hip_flexion_l',
            'hip_adduction_l',
            'hip_rotation_l',

            'hip_flexion_r',
            'hip_adduction_r',
            'hip_rotation_r',

            'knee_angle_r',
            'ankle_angle_r',
            'LTA',
            'RTA',
            'LRF',
            'RRF',
            'LBF',
            'RBF',
            'LGMED',
            'RGMED',
            'LMGAS',
            'RMGAS',
            'LVL',
            'RVL',
            'LGRAC',
            'RGRAC',
            'LGMAX',
            'RGMAX',
            'RShank_ACCX',
            'RShank_ACCY',
            'RShank_ACCZ',
            'RShank_GYROX',
            'RShank_GYROY',
            'RShank_GYROZ',
            'RAThigh_ACCX',
            'RAThigh_ACCY',
            'RAThigh_ACCZ',
            'RAThigh_GYROX',
            'RAThigh_GYROY',
            'RAThigh_GYROZ',
            'RPThigh_ACCX',
            'RPThigh_ACCY',
            'RPThigh_ACCZ',
            'RPThigh_GYROX',
            'RPThigh_GYROY',
            'RPThigh_GYROZ',
            'RPelvis_ACCX',
            'RPelvis_ACCY',
            'RPelvis_ACCZ',
            'RPelvis_GYROX',
            'RPelvis_GYROY',
            'RPelvis_GYROZ',
            'hip_flexion_velocity_l',
            'hip_adduction_velocity_l',
            'hip_rotation_velocity_l',

            'hip_flexion_velocity_r',
            'hip_adduction_velocity_r',
            'hip_rotation_velocity_r',
            'knee_velocity_r',
            'ankle_velocity_r',
            'subtalar_velocity_r',
        ]
        for df in self.dfs:
            for i in range(len(df) - window_size):
                for s in self.sensor_list:
                    torch.tensor(df[s])
                    feature = df['time'][i : i + window_size].drop(labels=[
                        'left',
                        'right',
                        'knee_angle_l_moment',
                        'ankle_angle_l_moment',
                        'subtalar_angle_l_moment',
                    ])
                    print(feature)
                    target = df['knee_angle_l_moment'][i : i + window_size]
                    print(target)
                    X.append(torch.tensor(feature))
                    y.append(torch.tensor(target))

    # returns number of samples in dataset
    def __len__(self):
        return self.len
    def calc_len(self):
        acc = 0
        for s in self.subjects:
            for a in self.activities[s]:
                acc += self.preprocessed_data[s][a][1]
        return acc
    # returns the idx sample
    def __getitem__(self, idx):
        acc = 0
        for s in self.subjects:
            for a in self.activities[s]:
                assert acc <= idx
                if idx < acc + self.preprocessed_data[s][a][1]:
                    df = pandas.read_csv(self.preprocessed_data[s][a][0], index_col="time")
                    window = df.iloc[idx - acc : idx - acc + self.window_size]
                    # I don't think pytorch accepts anything besides tensors
                    #sample_df = (window[config.sensor_list], s, a)
                    sample_df = window[config.sensor_list]
                    label_df = window[config.output_list]
                    # pytorch largely expects float32, not float64 which seems to be the numpy default
                    return torch.tensor(sample_df.to_numpy(dtype=np.float32)), torch.tensor(label_df.to_numpy(dtype=np.float32))
                else:
                    acc += self.preprocessed_data[s][a][1]
        #dy = self.oudataframe.iloc[idx, 0:]
        #return self.seq_len, dy
        assert(False)

class GreedyLSTMDataset(Dataset):
    def __init__(self, subjects, activities):
        #############################
        # initialize parameters
        self.window_size = config.window_size
        # get subjects and activities
        self.subjects = subjects
        self.activities = {}
        for s in self.subjects:
            self.activities[s] = [f for f in os.listdir(s) if activities.search(f)]
        self.preprocessed_data = {}
        for s in self.subjects:
            self.preprocessed_data[s] = {}
            for a in self.activities[s]:
                path = os.path.join(".", s, a, "preprocessed_data.csv")
                assert os.path.isfile(path)
                # keep all the data in memory
                df = pandas.read_csv(path, index_col="time")
                # if we're counting number of usable windows, it's +1
                num_windows = len(df.index) - self.window_size + 1
                self.preprocessed_data[s][a] = (df, num_windows)
        #############################
        self.len = self.calc_len()
        print(f"LSTMDataset has {self.len} windows")
        return
        #############################
    def __len__(self):
        return self.len
    def calc_len(self):
        acc = 0
        for s in self.subjects:
            for a in self.activities[s]:
                acc += self.preprocessed_data[s][a][1]
        return acc
    # returns the idx sample
    def __getitem__(self, idx):
        acc = 0
        for s in self.subjects:
            for a in self.activities[s]:
                assert acc <= idx
                if idx < acc + self.preprocessed_data[s][a][1]:
                    df = self.preprocessed_data[s][a][0]
                    window = df.iloc[idx - acc : idx - acc + self.window_size]
                    # I don't think pytorch accepts anything besides tensors
                    #sample_df = (window[config.sensor_list], s, a)
                    sample_df = window[config.sensor_list]
                    label_df = window[config.output_list]
                    ## we're only interested in the moments at the last timestamp
                    #label_df = df.iloc[idx - acc + self.window_size - 1]
                    #label_df = label_df[config.output_list]
                    # pytorch largely expects float32, not float64 which seems to be the numpy default
                    sample_t = torch.tensor(sample_df.to_numpy(dtype=np.float32))
                    label_t = torch.tensor(label_df.to_numpy(dtype=np.float32))
                    return sample_t, label_t
                else:
                    acc += self.preprocessed_data[s][a][1]
        #dy = self.oudataframe.iloc[idx, 0:]
        #return self.seq_len, dy
        assert(False)

if __name__ == '__main__':
    # initialize our dataloader
    subjects = [f for f in os.listdir('.') if re.search("AB\d+", f)]
    activities = re.compile(".");
    dataset = LSTMDataset(subjects, activities)
