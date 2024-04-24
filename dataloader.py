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

## our pytorch data loader
class GrandLSTMDataset(Dataset):
    def __init__(self, pickled_data, subjects, activities):
        with open(sys.argv[1], 'rb') as f:
            self.grandUnifiedData, self.windows = pickle.load(f)
        # if subjects or activities, windows needs to be corrected, by dropping
        # any windows that are for other subjects or activities
        self.windows = [(s, a, i) for s, a, i in self.windows if s in subjects and activities.search(a)]
        # save a some memory if we have lots of concurrent datasets
        # NOTE: yes this is repeated for every GrandLSTMDataset, but hopefully
        # it's not too slow
        for s in self.grandUnifiedData.keys():
            if s not in subjects:
                del self.grandUnifiedData[s]
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        s, a, i = self.windows[idx]
        window = self.grandUnifiedData[s][a].iloc[i : i + window_size]
        # I don't think pytorch accepts anything besides tensors
        sample_df, label_df = window[sensor_list], window[output_list]
        # pytorch largely expects float32, not float64 which seems to be the numpy default
        #sample_t = torch.tensor(sample_df.to_numpy(dtype=np.float32))
        #label_t = torch.tensor(label_df.to_numpy(dtype=np.float32))
        sample_t, label_t = torch.tensor(sample_df), torch.tensor(label_df)
        # these asserts should be redudant, since we should've filtered out any
        # windows with NaNs while generating grandUnifiedData
        assert not sample_t.isnan().any()
        assert not label_t.isnan().any()
        return sample_t, label_t
