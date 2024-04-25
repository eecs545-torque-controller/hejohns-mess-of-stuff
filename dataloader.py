import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas
import numpy as np
import pickle
import sys

from config import *

def filter_unused_data(subjects, activities, pickled_data):
    with open(pickled_data, 'rb') as f:
        grandUnifiedData, windows = pickle.load(f)
    # if subjects or activities, windows needs to be corrected, by dropping
    # any windows that are for other subjects or activities
    windows = [(s, a, i) for s, a, i in windows if s in subjects and activities.search(a)]
    # save some memory if we have lots of concurrent datasets
    # NOTE: yes this is repeated for every GrandLSTMDataset, but hopefully
    # it's not too slow
    for s in list(grandUnifiedData.keys()):
        if s not in subjects:
            del grandUnifiedData[s]
    return grandUnifiedData, windows

def get_window(grandUnifiedData, s, a, i):
    window = grandUnifiedData[s][a].iloc[i : i + window_size]
    # I don't think pytorch accepts anything besides tensors
    sample_df, label_df = window[sensor_list], window[output_list]
    # pytorch largely expects float32, not float64 which seems to be the numpy default
    sample_t, label_t = torch.tensor(sample_df.to_numpy(dtype=np.float32)), torch.tensor(label_df.to_numpy(dtype=np.float32))
    # these asserts should be redudant, since we should've filtered out any
    # windows with NaNs while generating GrandUnifiedData.pickle
    assert not sample_t.isnan().any()
    assert not label_t.isnan().any()
    return sample_t, label_t

def all_equal(l):
    return all(x == l[0] for x in l)

def normalize_data(grandUnifiedData):
    columnwise_sum = {}
    columnwise_count = {}
    columnwise_std = {}
    # I'd like to use columnwise pandas.DataFrame.sum, but this relies less on column order
    for c in sensor_list + output_list:
        columnwise_sum[c] = 0
        columnwise_count[c] = 0
        column_acc = []
        for s in grandUnifiedData.keys():
            for a in grandUnifiedData[s].keys():
                column_acc.extend(grandUnifiedData[s][a][c].to_list())
        cs = pandas.Series(column_acc)
        columnwise_sum[c] = cs.sum()
        columnwise_count[c] = len(column_acc)
        columnwise_std[c] = cs.std()

        # overkill for GreedyGrandLSTMDataset, but whatever
        for s in grandUnifiedData.keys():
            for a in grandUnifiedData[s].keys():
                dfc = grandUnifiedData[s][a][c]
                grandUnifiedData[s][a][c] = (dfc - (columnwise_sum[c] / columnwise_count[c])) / columnwise_std[c]
    assert all_equal([num for c, num in columnwise_count])
    return grandUnifiedData

## our pytorch data loader
class GrandLSTMDataset(Dataset):
    def __init__(self, subjects, activities, pickled_data="GrandUnifiedData.pickle"):
        grandUnifiedData, self.windows = filter_unused_data(subjects, activities, pickled_data)
        self.grandUnifiedData = normalize_data(grandUnifiedData)
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s, a, i = self.windows[idx]
        return get_window(self.grandUnifiedData, s, a, i)

# precompute all windows and try to fit them all in memory
class GreedyGrandLSTMDataset(Dataset):
    def __init__(self, subjects, activities, pickled_data="GrandUnifiedData.pickle"):
        grandUnifiedData, windows = filter_unused_data(subjects, activities, pickled_data)
        grandUnifiedData = normalize_data(grandUnifiedData)
        self.windows = [get_window(grandUnifiedData, s, a, i) for s, a, i in windows]
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx]
