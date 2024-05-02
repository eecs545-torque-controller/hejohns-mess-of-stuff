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
    test_subjects = ['AB01']
    #if DEBUG:
    #    activities = re.compile("normal_walk_1_0-6"); # smaller dataset
    #else:
    activities = re.compile(".");
    test_data = dataloader.GrandLSTMDataset(window_size, (grandUnifiedData, windows), test_subjects, activities)
    test_dataloader = torch.utils.data.DataLoader(
            test_data,
            shuffle=True,
            batch_size=batch_size,
            num_workers=32,
            persistent_workers=True,
            )
    assert len(sys.argv) > 2 and os.path.isfile(sys.argv[2])
    checkpoint = torch.load(sys.argv[2], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

def joint_total_mse(dataloader, joint):
    assert not model.training
    total_loss, num_elements = loop_over_data(model, dataloader, loss_fn=nn.MSELoss(reduction="none"), y_lambda=lambda y_pred, y_batch: (y_pred[:,joint], y_batch[:,joint]))
    return total_loss, num_elements

def joint_eval_rmse(dataloader, joint):
    assert not model.training
    total_loss, num_elements = joint_total_mse(dataloader, joint)
    return rmse(total_loss, num_elements)

if __name__ == '__main__':
    model.eval()
    with torch.no_grad():
        for i in range(len(output_list)):
            joint_rmse = joint_eval_rmse(test_dataloader, i)
            print(f"{output_list[i]} {joint_rmse}")
