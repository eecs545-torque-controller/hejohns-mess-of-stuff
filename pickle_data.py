#!/usr/bin/env python3

import pandas
import sys
import os
import pickle

assert len(sys.argv) == 2
subjects = [f for f in os.listdir(os.getcwd()) if re.search("^AB\d+$", f)]
activities_re = re.compile(".")
grandUnifiedData = {}
activities = {}
for s in subjects:
    grandUnifiedData[s] = {}
    activities[s] = [f for f in os.listdir(os.path.join(os.getcwd, s)) if activities_re.search(f)]
    for a in activities[s]:
        leaf = os.path.join(os.getcwd(), s, a, sys.argv[1])
        assert os.path.isfile(leaf)
        df = pandas.read_csv(leaf, index_col="time")
        grandUnifiedData[s][a] = df
pickle.dump(grandUnifiedData, sys.stdout)
