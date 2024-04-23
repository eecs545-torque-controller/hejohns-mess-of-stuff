import pandas
import pickle
import sys
import numpy as np

from config import *

assert len(sys.argv) == 3
with open(sys.argv[1], 'rb') as f:
    grandUnifiedData = pickle.load(f)
with open(sys.argv[2], 'wb') as g:
    for s, v in grandUnifiedData.items():
        for a, df in v.items():
            # https://stackoverflow.com/a/69188251
            df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
            for i in range(len(df.index) - window_size):
                dg = df.iloc[i : i + window_size]
                if dg.isnull().values.any():
                    continue
                else:
                    pickle.dump(dg, g)
