#!/usr/bin/env perl

# wrapper around join.py, for each subject and activity

use v5.36;
use utf8;
use strict;
use warnings;

use File::Slurp;
use Carp::Assert;
use Parallel::Loops;

my $parallel = Parallel::Loops->new(8);

my @subjects = sort(grep {/AB\d+/} File::Slurp::read_dir('.'));
$parallel->foreach(\@subjects, sub {
    my $s = $_;
    my @activities = sort(File::Slurp::read_dir("./$s/"));
    for my $a (@activities){
        #say "./$s/$a/";
        my @data = sort(grep {/(activity_flag|moment_filt|angle|emg_downsampled|imu_real|velocity)\.csv$/} File::Slurp::read_dir("./$s/$a"));
        @data = map {"./$s/$a/$_"} @data;
        my @activity = grep {/activity_flag\.csv$/} @data;
        assert(@activity == 1);
        my @data_without_activity = grep {!/\Q$activity[0]\E/} @data;
        #say "@{['python3', 'join.py', @activity, @data_without_activity]}";
        system('python3', 'join.py', @activity, @data_without_activity);
        assert($? == 0);
    }
});
