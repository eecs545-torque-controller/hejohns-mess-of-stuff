#!/usr/bin/env python3

# downsample a single emg csv file, with filename sys.argv[1]

import pandas
import sys
import os.path

# rewrite of Zach's script
df = pandas.read_csv(sys.argv[1], index_col='time')
#print(df)
df = df.abs()
#print(df)
moving_avg = df.rolling(10, min_periods=1)
df = moving_avg.mean()
# downsample by selecting every 10th row
df = df.iloc[::10]
# NOTE: we'd have to drop one timestamp for every other sensor if we did something else, so let's just do the simple thing
#print(df)
#shifted_index = df.index - 0.0045
#df.reindex(shifted_index)
#print(df)
d = os.path.dirname(os.path.normpath(sys.argv[1]))
df.to_csv(os.path.join(d, f"emg_downsampled.csv"))
