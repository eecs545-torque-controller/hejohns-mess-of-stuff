#!/usr/bin/env python3

import pandas
import sys

df = pandas.read_csv(sys.argv[1], index_col='time')
keep = (df['left'] | df['right'])
keep = keep.astype('bool')
df = df.loc[keep]
df.to_csv(sys.stdout)
