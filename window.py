import pandas
import pickle
import sys
import numpy as np

assert len(sys.argv) == 2
grandUnifiedData = pickle.load(sys.argv[1])
windowedData = []
# let's hope that keeping grandUnifiedData in memory isn't an issue
for s, v in grandUnifiedData.items():
    for a, df in v.items():
        # https://stackoverflow.com/a/69188251
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
        for i in range(len(df.index) - window_size):
            dg = df.iloc[i : i + window_size]
            if dg.isnull().values.any():
                continue
            else:
                windowedData.append(dg)
with open(sys.argv[2], 'wb') as f:
    pickle.dump(windowedData, f)
