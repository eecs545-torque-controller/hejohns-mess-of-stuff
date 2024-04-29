#!/usr/bin/env python3

import torch
import datetime
import pytz
import pickle
import os
import re

# TODO: check this again
sensor_list = [
        'hip_flexion_l', # good
        #'hip_adduction_l', # unused?
        #'hip_rotation_l', # unused?

        'hip_flexion_r', # good
        #'hip_adduction_r', # unused?
        #'hip_rotation_r', # unused?

        'knee_angle_r', # good
        'ankle_angle_r', # good
        #'subtalar_angle_r', # unused?

        #'LTA', # unused?
        'RTA', # good
        'LRF', # good
        'RRF', # good
        'LBF', # good
        'RBF', # good
        'LGMED', # good
        'RGMED', # good
        'LMGAS',
        'RMGAS', # good
        'LVL', # good
        'RVL', # good
        'LGRAC', # good
        'RGRAC', # good
        'LGMAX', # good
        'RGMAX', # good

        'RShank_ACCX', # good
        'RShank_ACCY', # good
        'RShank_ACCZ', # good
        'RShank_GYROX', # good
        'RShank_GYROY', # good
        'RShank_GYROZ', # good

        'RAThigh_ACCX', # good
        'RAThigh_ACCY', # good
        'RAThigh_ACCZ', # good
        'RAThigh_GYROX', # good
        'RAThigh_GYROY', # good
        'RAThigh_GYROZ', # good

        'RPThigh_ACCX', # good
        'RPThigh_ACCY', # good
        'RPThigh_ACCZ', # good
        'RPThigh_GYROX', # good
        'RPThigh_GYROY', # good
        'RPThigh_GYROZ', # good

        'LAThigh_ACCX', # good
        'LAThigh_ACCY', # good
        'LAThigh_ACCZ', # good
        'LAThigh_GYROX', # good
        'LAThigh_GYROY', # good
        'LAThigh_GYROZ', # good

        'LPThigh_ACCX', # good
        'LPThigh_ACCY', # good
        'LPThigh_ACCZ', # good
        'LPThigh_GYROX', # good
        'LPThigh_GYROY', # good
        'LPThigh_GYROZ', # good

        'LPelvis_ACCX', # good
        'LPelvis_ACCY', # good
        'LPelvis_ACCZ', # good
        'LPelvis_GYROX', # good
        'LPelvis_GYROY', # good
        'LPelvis_GYROZ', # good

        'RPelvis_ACCX', # good
        'RPelvis_ACCY', # good
        'RPelvis_ACCZ', # good
        'RPelvis_GYROX', # good
        'RPelvis_GYROY', # good
        'RPelvis_GYROZ', # good

        'hip_flexion_velocity_l', # good
        #'hip_adduction_velocity_l',
        #'hip_rotation_velocity_l',

        'hip_flexion_velocity_r', # good
        #'hip_adduction_velocity_r',
        #'hip_rotation_velocity_r',
        'knee_velocity_r', # good
        'ankle_velocity_r', # good
        #'subtalar_velocity_r', # unused
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
