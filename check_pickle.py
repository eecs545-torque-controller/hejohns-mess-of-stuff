#!/usr/bin/env python3

import pandas
import pickle
import os
import re

from config import *
from dataloader import *

subjects = [f for f in os.listdir('.') if re.search("AB\d+", f)]
activities = re.compile(".");
dataset = GrandLSTMDataset(subjects, activities)
for s in dataset.grandUnifiedData.keys():
    for a in dataset.grandUnifiedData[s].keys():
        print(f"{s}/{a}")
        print(dataset.grandUnifiedData[s][a])
print("--------------------")
for w in dataset.windows:
    print(w)

