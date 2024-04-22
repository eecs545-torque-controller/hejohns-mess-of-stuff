#!/usr/bin/env python3

# join all the sensor data files sys.argv[2:] together, keyed by time, and
# output the aggregate file in dir sys.argv[1]

import pandas
import sys
import os.path

df = pandas.read_csv(sys.argv[1], index_col="time")
#print(df)
for f in sys.argv[2:]:
    df2 = pandas.read_csv(f, index_col="time")
    #print(df2)
    df = df.join(df2) # on index
d = os.path.dirname(sys.argv[1])
df.to_csv(os.path.join(d, f"aggregate.csv"))
