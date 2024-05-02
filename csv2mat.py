#!/usr/bin/env python3

import pandas
import scipy.io
import sys

df = pandas.read_csv(sys.argv[1])
scipy.io.savemat(sys.argv[2], {"ans": df.to_dict()})
