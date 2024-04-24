#!/usr/bin/env python3

# pickle entire dataset as grandUnifiedData
# and store indices for windows in windows

import pandas
import sys
import os
import pickle
import re
import numpy as np

from config import *

assert len(sys.argv) == 3
subjects = [f for f in os.listdir(os.getcwd()) if re.search("^AB\d+$", f)]
activities_re = re.compile(".")
grandUnifiedData = {}
windows = []
for s in subjects:
    grandUnifiedData[s] = {}
    activities = [f for f in os.listdir(os.path.join(os.getcwd(), s)) if activities_re.search(f)]
    for a in activities:
        # get dataframe for this [subject][activity]
        leaf = os.path.join(os.getcwd(), s, a, sys.argv[1])
        assert os.path.isfile(leaf)
        df = pandas.read_csv(leaf, index_col="time")
        # don't need these anymore, since we've already filtered
        df = df.drop(labels=['left', 'right'], axis=1)
        # https://stackoverflow.com/a/69188251
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
        grandUnifiedData[s][a] = df
        # calculate window indices
        for i in range(len(df.index) - window_size):
            df = df[sensor_list + output_list]
            dg = df.iloc[i : i + window_size]
            if not dg.isnull().values.any():
                windows.append((s, a, i))
with open(sys.argv[2], 'wb') as f:
    pickle.dump((grandUnifiedData, windows), f)
