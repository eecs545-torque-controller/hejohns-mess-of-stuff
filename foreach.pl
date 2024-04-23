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
assert(@ARGV == 3, "$0 scriptname sensorfile outputfile");
my $parallel = Parallel::Loops->new($ncpus);
my @subjects = sort(grep {/^AB\d+$/} read_dir(curdir()));
$parallel->foreach(\@subjects, sub {
    my $s = $_;
    my @activities = sort(read_dir(catdir(curdir(), $s)));
    for my $a (@activities){
        my @data = sort(grep {/\Q$ARGV[1]\E\.csv$/} File::Slurp::read_dir(catdir(curdir(), $s, $a)));
        assert(@data == 1, "Trying to get $ARGV[1], but got @data");
        if(!-e catfile(curdir(), $s, $a, $ARGV[2])){
            run ['python3', $ARGV[0], catfile(curdir(), $s, $a, $data[1])],
                '>', catfile(curdir(), $s, $a, $ARGV[2])
                or die "$ARGV[0] failed!";
        }
    }
});
