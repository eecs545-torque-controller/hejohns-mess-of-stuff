#!/usr/bin/env python3

import pandas
import sys
# our files
from config import *

def normalize_data(grandUnifiedData):
    columnwise_sum = {}
    columnwise_count = {}
    columnwise_std = {}
    # I'd like to use columnwise pandas.DataFrame.sum, but this relies less on column order
    for c in sensor_list:
        columnwise_sum[c] = 0
        columnwise_count[c] = 0
        column_acc = []
        for s in grandUnifiedData.keys():
            for a in grandUnifiedData[s].keys():
                column_acc.extend(grandUnifiedData[s][a][c].to_list())
        cs = pandas.Series(column_acc)
        columnwise_sum[c] = cs.sum()
        columnwise_count[c] = len(column_acc)
        columnwise_std[c] = cs.std()

        for s in grandUnifiedData.keys():
            for a in grandUnifiedData[s].keys():
                dfc = grandUnifiedData[s][a][c]
                grandUnifiedData[s][a][c] = (dfc - (columnwise_sum[c] / columnwise_count[c])) / columnwise_std[c]
    assert all_equal([columnwise_count[c] for c in columnwise_count.keys()])
    return grandUnifiedData, columnwise_sum, columnwise_count, columnwise_std

assert len(sys.argv) == 3
grandUnifiedData, windows, *_ = read_entire_pickle(sys.argv[1])
grandUnifiedData, columnwise_sum, columnwise_count, columnwise_std = normalize_data(grandUnifiedData)
with open(sys.argv[2], 'wb') as f:
    pickle.dump((grandUnifiedData, windows, columnwise_sum, columnwise_count, columnwise_std), f)
