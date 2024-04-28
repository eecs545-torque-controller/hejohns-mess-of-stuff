#!/usr/bin/env python3

import pandas
import pickle
import os
import re

from config import *
from dataloader import *

def check(filepath):
    grandUnifiedData, windows, *normalization_params = read_entire_pickle(filepath=filepath)
    print(f"has {len(windows)} windows")
    print("normalization_params: ", normalization_params)
    subjects = grandUnifiedData.keys()
    activities = re.compile(".");
    dataset = GrandLSTMDataset((grandUnifiedData, windows), subjects, activities)
    print("--------------------")
    for s in dataset.grandUnifiedData.keys():
        for a in dataset.grandUnifiedData[s].keys():
            print(f"{s}/{a}")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                print(dataset.grandUnifiedData[s][a])
    print("--------------------")
    for w in dataset.windows:
        print(w)
    print("--------------------")
    
check("GrandUnifiedData.pickle")
print("====================")
check("GrandUnifiedData_normalized.pickle")
