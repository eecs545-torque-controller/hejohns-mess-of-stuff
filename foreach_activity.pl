#!/usr/bin/env perl

# wrapper to call the python script $ARGV[0] for each datafile matching $ARGV[1] exactly?? Why??

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
        my @data = sort(grep {/\Q$ARGV[1]\E\.csv$/} File::Slurp::read_dir("./$s/$a"));
        assert(@data == 1);
        #say "./$s/$a/$data[0]";
        system('python3', $ARGV[0], "./$s/$a/$data[0]");
        assert($? == 0);
    }
});
