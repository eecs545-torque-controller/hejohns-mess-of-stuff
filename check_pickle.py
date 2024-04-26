#!/usr/bin/env python3

import pandas
import pickle
import os
import re

from config import *
from dataloader import *

pickled_data = read_entire_pickle()
subjects = pickled_data[0].keys()
activities = re.compile(".");
dataset = GrandLSTMDataset(pickled_data, subjects, activities)
for s in dataset.grandUnifiedData.keys():
    for a in dataset.grandUnifiedData[s].keys():
        print(f"{s}/{a}")
        print(dataset.grandUnifiedData[s][a])
print("--------------------")
for w in dataset.windows:
    print(w)

