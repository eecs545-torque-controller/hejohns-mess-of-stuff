#!/usr/bin/env python3

import pandas
import sys
import os.path

df = pandas.read_csv(sys.argv[1], index_col='time')
#print(df)
keep = (df['left'] | df['right'])
keep = keep.astype('bool')
#print(keep)
df = df.loc[keep]
#print(df)
d = os.path.dirname(os.path.normpath(sys.argv[1]))
df.to_csv(os.path.join(d, f"filtered.csv"))
