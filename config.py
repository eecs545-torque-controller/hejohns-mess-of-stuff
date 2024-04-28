#!/usr/bin/env python3

import torch
import datetime
import pytz
import pickle
import os
import re

# TODO: check this again
sensor_list = [
        'hip_flexion_l',
        'hip_adduction_l',
        'hip_rotation_l',

        'hip_flexion_r',
        'hip_adduction_r',
        'hip_rotation_r',

        'knee_angle_r',
        'ankle_angle_r',
        'subtalar_angle_r',

        'LTA',
        'RTA',
        'LRF',
        'RRF',
        'LBF',
        'RBF',
        'LGMED',
        'RGMED',
        'LMGAS',
        'RMGAS',
        'LVL',
        'RVL',
        'LGRAC',
        'RGRAC',
        'LGMAX',
        'RGMAX',
        'RShank_ACCX',
        'RShank_ACCY',
        'RShank_ACCZ',
        'RShank_GYROX',
        'RShank_GYROY',
        'RShank_GYROZ',
        'RAThigh_ACCX',
        'RAThigh_ACCY',
        'RAThigh_ACCZ',
        'RAThigh_GYROX',
        'RAThigh_GYROY',
        'RAThigh_GYROZ',
        'RPThigh_ACCX',
        'RPThigh_ACCY',
        'RPThigh_ACCZ',
        'RPThigh_GYROX',
        'RPThigh_GYROY',
        'RPThigh_GYROZ',
        'RPelvis_ACCX',
        'RPelvis_ACCY',
        'RPelvis_ACCZ',
        'RPelvis_GYROX',
        'RPelvis_GYROY',
        'RPelvis_GYROZ',
        'hip_flexion_velocity_l',
        'hip_adduction_velocity_l',
        'hip_rotation_velocity_l',

        'hip_flexion_velocity_r',
        'hip_adduction_velocity_r',
        'hip_rotation_velocity_r',
        'knee_velocity_r',
        'ankle_velocity_r',
#        'subtalar_velocity_r',
        ]

output_list = [
        'knee_angle_l_moment',
        'ankle_angle_l_moment',
# NOTE: NOT USING THIS
#        'subtalar_angle_l_moment',
        ]

batch_size = 64

# this should really be part of the binary data of the data pickle file,
# but I don't want to recompute 50 yet again and change the order
window_size_re = re.compile("\.(\d+)\.pickle$")
def get_window_size(grandUnifiedDataPath):
    window_size = int(window_size_re.search(grandUnifiedDataPath).group(1))
    print(f"Using a window_size of {window_size}")
    return window_size

device = "cuda" if torch.cuda.is_available() else "cpu"

n_epochs = 4000 # taken from the paper

def read_entire_pickle(filepath):
    with open(filepath, 'rb') as f:
        grandUnifiedData, windows, *normalization_params = pickle.load(f)
    return grandUnifiedData, windows, normalization_params

def all_equal(l):
    return all(x == l[0] for x in l)

# little performance debugging helper
def curtime():
    return datetime.datetime.now(tz=pytz.timezone('US/Eastern')).time()

# if we should only predict the moments at the last timestamp, or the entire window
LAST = False if os.environ.get("WHOLE_WINDOW") else True

DEBUG = True if os.environ.get("DEBUG") else False

SCHEDULER = True if os.environ.get("SCHEDULER") else False

if __name__ == '__main__':
    print("This file shouldn't be run directly")
