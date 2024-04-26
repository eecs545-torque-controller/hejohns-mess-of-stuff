#!/usr/bin/env python3

import pandas
import pickle
import os
import re

from config import *
from dataloader import *

def check(filepath):
    grandUnifiedData, windows, *normalization_params = read_entire_pickle(filepath=filepath)
    print(f"dataset has {len(dataset.windows)} windows")
    print("normalization_params: ", normalization_params)
    subjects = grandUnifiedData.keys()
    activities = re.compile(".");
    dataset = GrandLSTMDataset(pickled_data, subjects, activities)
    print("--------------------")
    for s in dataset.grandUnifiedData.keys():
        for a in dataset.grandUnifiedData[s].keys():
            print(f"{s}/{a}")
            print(dataset.grandUnifiedData[s][a])
    print("--------------------")
    for w in dataset.windows:
        print(w)
    print("--------------------")
    
check("GrandUnifiedData.pickle")
check("GrandUnifiedData_normalized.pickle")
