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
#class GrandLSTMDataset(Dataset):
#    def __init__(self, pickled_data, subjects, activities):
#    def __len__(self):
#    def __getitem__(self, idx):
# https://stackoverflow.com/a/12762056
with open(sys.argv[1], 'rb') as f:
    grandUnifiedData, windows = pickle.load(f)
print(len(windows))

with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(grandUnifiedData[windows[-1][0]][windows[-1][1]].iloc[windows[-1][2] : windows[-1][2] + window_size])
#for s in subjects:
#    for a in activities[s]:
#        grandUnifiedData[s][a][sensor_list]
#        grandUnifiedData[s][a][output_list]
