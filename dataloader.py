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

def filter_windows(windows, subjects, activities):
    # if subjects or activities, windows needs to be corrected, by dropping
    # any windows that are for other subjects or activities
    return [(s, a, i) for s, a, i in windows if s in subjects and activities.search(a)]

def get_window(grandUnifiedData, s, a, i):
    window = grandUnifiedData[s][a].iloc[i : i + window_size]
    # I don't think pytorch accepts anything besides tensors
    sample_df, label_df = window[sensor_list], window[output_list]
    if LAST:
        # we could do this with [-1, :] after converting to a tensor, but let's do it even before it gets there
        label_df = label_df.loc[label_df.index[-1]].squeeze()
    # pytorch by default expects float32, not float64 which seems to be the numpy default
    sample_t, label_t = torch.tensor(sample_df.to_numpy(dtype=np.float32)), torch.tensor(label_df.to_numpy(dtype=np.float32))
    # these asserts should be redudant, since we should've filtered out any
    # windows with NaNs while generating GrandUnifiedData.pickle
    assert not sample_t.isnan().any()
    assert not label_t.isnan().any()
    return sample_t, label_t

# our pytorch data loader
class GrandLSTMDataset(Dataset):
    def __init__(self, pickled_data, subjects, activities):
        self.grandUnifiedData, windows = pickled_data
        print(f"starting to filter_windows... {curtime()}")
        self.windows = filter_windows(windows, subjects, activities)
        print(f"filter_windows finished at {curtime()}")
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s, a, i = self.windows[idx]
        return get_window(self.grandUnifiedData, s, a, i)

# precompute all windows and try to fit them all in memory
# NOTE: requires ~46G memory on full dataset, ~2G on just one subject, norm_walk*
class GreedyGrandLSTMDataset(Dataset):
    def __init__(self, pickled_data, subjects, activities):
        grandUnifiedData, windows = pickled_data
        print(f"starting to filter_windows... {curtime()}")
        windows = filter_windows(windows, subjects, activities)
        print(f"filter_windows finished at {curtime()}")
        print(f"starting to get_window ... {curtime()}")
        self.unused_data = {}
        if len(windows) > 1000000:
            # serializing grandUnifiedData is REALLY slow, so it only makes sense to do when the number of windows is very large
            ## first, let's try to cut down on size of grandUnifiedData
            for s in list(grandUnifiedData.keys()):
                if s not in subjects:
                    self.unused_data[s] = grandUnifiedData[s]
                    del grandUnifiedData[s]
            ## and now use multiprocess.map
            nproc = min(len(os.sched_getaffinity(0)), 8)
            with Pool(processes=nproc) as pool:
                self.windows = pool.map((lambda t: get_window(grandUnifiedData, t[0], t[1], t[2])), windows)
        else:
            self.windows = [get_window(grandUnifiedData, s, a, i) for s, a, i in windows]
        print(f"get_window finished at {curtime()}", flush=True)
        # NOTE: drop all the data we're using, since no other datasets should be using it
        # as a memory optimization (the test dataset doesn't need the entire dateset, only what's left)
        for s in list(grandUnifiedData.keys()):
            if s in subjects:
                del grandUnifiedData[s]
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx]
