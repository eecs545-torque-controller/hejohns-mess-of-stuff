#!/usr/bin/env python3

import pandas
import sys
import os
import pickle
import re
import numpy as np

assert len(sys.argv) == 3
subjects = [f for f in os.listdir(os.getcwd()) if re.search("^AB\d+$", f)]
activities_re = re.compile(".")
grandUnifiedData = {}
for s in subjects:
    grandUnifiedData[s] = {}
    activities = [f for f in os.listdir(os.path.join(os.getcwd(), s)) if activities_re.search(f)]
    for a in activities:
        leaf = os.path.join(os.getcwd(), s, a, sys.argv[1])
        assert os.path.isfile(leaf)
        df = pandas.read_csv(leaf, index_col="time")
        # don't need these anymore, since we've already filtered
        df = df.drop(labels=['left', 'right'], axis=1)
        # https://stackoverflow.com/a/69188251
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
        grandUnifiedData[s][a] = df
with open(sys.argv[2], 'wb') as f:
    pickle.dump(grandUnifiedData, f)
