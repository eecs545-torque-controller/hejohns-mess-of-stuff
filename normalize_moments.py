#!/usr/bin/env python3

import pandas
import sys
import re
# our files
from config import *

assert len(sys.argv) == 3
grandUnifiedData, windows, _ = read_entire_pickle(sys.argv[1])
masses_df = pandas.read_csv("subject_mass.csv", index_col="subject")
for s in grandUnifiedData.keys():
    for a in grandUnifiedData[s].keys():
        df = grandUnifiedData[s][a]
        m = re.compile("moment$")
        moment_columns = [ c for c in df.columns.tolist() if m.search(c)]
        df[moment_columns] = df[moment_columns] / masses_df.at[s, "kg"]
        grandUnifiedData[s][a] = df # I think this is redundant
with open(sys.argv[2], 'wb') as f:
    pickle.dump((grandUnifiedData, windows), f)
