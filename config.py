#!/usr/bin/env python3

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
        'subtalar_velocity_r',
        ]

output_list = [
        'knee_angle_l_moment',
        'ankle_angle_l_moment',
        'subtalar_angle_l_moment',
        ]

batch_size = 64

window_size = 50

if __name__ == '__main__':
    print("This file shouldn't be run directly")
