#!/usr/bin/env perl

use v5.36;
use utf8;
use strict;
use warnings;

use File::Slurp;
use Carp::Assert;
use JSON;
use Hash::Subset;

my %total_data = %{JSON::decode_json (do {
    local $/ = undef;
    <STDIN>;
})};
for my $s(keys %total_data){
    for my $a (keys $total_data{$s}->%*){
        for my $t (keys $total_data{$s}{$a}->%*){
            $total_data{$s}{$a}{$t} = Hash::Subset::hash_subset($total_data{$s}{$a}{$t},
                [
                    'left',
                    'right',
                    'knee_angle_l_moment',
                    'ankle_angle_l_moment',
                    'subtalar_angle_l_moment',
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
            );
        }
    }
}
print JSON::to_json(\%total_data, {canonical => 1});
