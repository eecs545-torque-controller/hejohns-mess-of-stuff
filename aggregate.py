#!/usr/bin/env python3

# join all the sensor data files sys.argv[1:] together, keyed by time, and
# print the aggregate file

import pandas
import sys

df = pandas.read_csv(sys.argv[1], index_col="time")
for f in sys.argv[2:]:
    df2 = pandas.read_csv(f, index_col="time")
    df = df.join(df2) # on index
df.to_csv(sys.stdout)
