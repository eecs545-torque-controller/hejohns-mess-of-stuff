#!/usr/bin/env python3

# Because I'm an idiot, we have to do this
# THIS IS OVER NORMALIZED DATA

import pandas
import sys
import os
import pickle
import re
import numpy as np

from config import *

assert len(sys.argv) == 3

grandUnifiedData, _windows, normalization_params = read_entire_pickle(sys.argv[1])
window_size = get_window_size(sys.argv[2])
windows = []
for s in grandUnifiedData.keys():
    for a in grandUnifiedData[a].keys():
        df = grandUnifiedData[s][a]
        for i in range(len(df.index) - window_size):
            if not df.iloc[i : i + window_size].isnull().values.any():
                windows.append((s, a, i))
with open(sys.argv[2], 'wb') as f:
    pickle.dump((grandUnifiedData, windows) + tuple(normalization_params), f)
