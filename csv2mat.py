#!/usr/bin/env python3

import pandas
import scipy.io
import sys

df = pandas.read_csv(sys.argv[1])
data = df.to_dict("list")
scipy.io.savemat(sys.argv[2], {"ans": data})
