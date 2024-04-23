#!/usr/bin/env perl

use v5.36;
use utf8;
use strict;
use warnings;

use File::Slurp qw(read_file);
use File::Spec::Functions;
use Carp::Assert;
use Parallel::Loops;
use IPC::Run qw(run);

my $ncpus;
chomp($ncpus = `nproc`);
$ncpus /= 4;
my $parallel = Parallel::Loops->new($ncpus);
my @subjects = sort grep {/^AB\d+$/} read_dir('.');
$parallel->foreach(\@subjects, sub {
    my $s = $_;
    my @activities = sort read_dir(catdir(curdir(), $s));
    for my $a (@activities){
        my @data = sort grep {/(activity_flag|moment_filt|angle|emg_downsampled|imu_real|velocity)\.csv$/} read_dir(catdir(curdir(), $s, $a));
        @data = map {catfile(curdir(), $s, $a, $_)} @data;
        my @activity = grep {/activity_flag\.csv$/} @data;
        assert(@activity == 1);
        my @data_without_activity = grep {!/\Q$activity[0]\E/} @data;
        my $python_script = $0;
        $python_script =~ s/\.\S+$/.py/;
        assert(defined($ARGV[0])):
        run ['python3', $python_script, @activity, @data_without_activity],
            '>', catfile(curdir(), $s, $a, $ARGV[0])
            or die "$python_script failed!";
    }
});
