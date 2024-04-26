import torch
from torch.utils.data import Dataset
import pandas
import numpy as np
import pickle
import sys
import os
from multiprocess import Pool
import dill # required by multiprocess, although we're not actually using it directly

from config import *

def filter_unused_data(windows, subjects, activities):
    # if subjects or activities, windows needs to be corrected, by dropping
    # any windows that are for other subjects or activities
    return [(s, a, i) for s, a, i in windows if s in subjects and activities.search(a)]

def get_window(grandUnifiedData, s, a, i):
    window = grandUnifiedData[s][a].iloc[i : i + window_size]
    # I don't think pytorch accepts anything besides tensors
    sample_df, label_df = window[sensor_list], window[output_list]
    # pytorch by default expects float32, not float64 which seems to be the numpy default
    sample_t, label_t = torch.tensor(sample_df.to_numpy(dtype=np.float32)), torch.tensor(label_df.to_numpy(dtype=np.float32))
    # these asserts should be redudant, since we should've filtered out any
    # windows with NaNs while generating GrandUnifiedData.pickle
    assert not sample_t.isnan().any()
    assert not label_t.isnan().any()
    return sample_t, label_t

def all_equal(l):
    return all(x == l[0] for x in l)

# really slow algorithm but only run on init
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
    assert all_equal([columnwise_count[c] for c in columnwise_count.keys()])
    return grandUnifiedData, columnwise_sum, columnwise_count, columnwise_std

# our pytorch data loader
class GrandLSTMDataset(Dataset):
    def __init__(self, normalized_data, subjects, activities):
        print(f"starting to filter_unused_data... {curtime()}")
        self.normalized_data, windows = normalized_data
        self.windows = filter_unused_data(windows, subjects, activities)
        print(f"filter_unused_data finished at {curtime()}")
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s, a, i = self.windows[idx]
        return get_window(self.normalized_data, s, a, i)

# precompute all windows and try to fit them all in memory
# NOTE: requires ~46G memory on full dataset, ~2G on just one subject, norm_walk*
class GreedyGrandLSTMDataset(Dataset):
    def __init__(self, normalized_data, subjects, activities):
        normalized_data, windows = normalized_data
        print(f"starting to filter_unused_data... {curtime()}")
        windows = filter_unused_data(windows, subjects, activities)
        print(f"filter_unused_data finished at {curtime()}")
        print(f"starting to get_window ... {curtime()}")
        with Pool(processes=min(len(os.sched_getaffinity(0)), 8)) as pool:
            #self.windows = [get_window(grandUnifiedData, s, a, i) for s, a, i in windows]
            self.windows = pool.map((lambda t: get_window(normalized_data, t[0], t[1], t[2])), windows)
        print(f"get_window finished at {curtime()}", flush=True)
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx]
