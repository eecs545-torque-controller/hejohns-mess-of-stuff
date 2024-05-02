#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas
import numpy as np
import sys
import os
import re
import time
import copy
# our files
from config import *
import dataloader
from model import *

if __name__ == '__main__':
    # gpu stuff
    print(f"Running on {device}")
    # args
    assert len(sys.argv) > 1
    window_size = get_window_size(sys.argv[1])
    use_greedy = window_size < 50
    print(f"use_greedy is {use_greedy}")
    grandUnifiedData, windows, normalization_params = read_entire_pickle(sys.argv[1])
    # basic initialization
    model = LSTMModel()
    model = nn.DataParallel(model)
    model = model.to(device, non_blocking=True)
    #if DEBUG:
    #    subjects = ['AB01', 'AB02', 'AB05']
    #else:
    test_subjects = ['AB05']
    #if DEBUG:
    #    activities = re.compile("normal_walk_1_0-6"); # smaller dataset
    #else:
    #activities = re.compile("normal_walk_1_2-5");
    activities = re.compile("normal_walk_1_shuffle");
    test_data = dataloader.GrandLSTMDataset(window_size, (grandUnifiedData, windows), test_subjects, activities)
    assert len(sys.argv) > 2 and os.path.isfile(sys.argv[2])
    checkpoint = torch.load(sys.argv[2], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

if __name__ == '__main__':
    model.eval()
    print(f",timestep,knee,ankle")
    with torch.no_grad():
        # in a rush-- let's just copy and paste loop_over_data
        for (s, a, j) in test_data.windows:
            file = s + "/" + a + "/" + "preprocessed_data.csv"
            df = pandas.read_csv(file, index_col="time")
            df = df.iloc[j:j + window_size]
            df = df[sensor_list]
            dft = torch.tensor(df.to_numpy(dtype=np.float32)).unsqueeze(0)
            y_pred = model(dft)
            assert not y_pred.isnan().any()
            y_pred = y_pred.squeeze()
            # TODO: uhhhh j + window_size ?
            print(f"{j + window_size},{df.index.values[-1]},{y_pred[0]},{y_pred[1]}")
