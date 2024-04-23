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
my @subjects = sort(grep {/^AB\d+$/} read_dir(curdir()));
$parallel->foreach(\@subjects, sub {
    my $s = $_;
    my @activities = sort(read_dir(catdir(curdir(), $s)));
    for my $a (@activities){
        my @data = sort(grep {/emg\.csv$/} File::Slurp::read_dir(catdir(curdir(), $s, $a)));
        assert(@data == 1, "Trying to get emg data, but got @data");
        my $python_script = $0;
        $python_script =~ s/\.\S+$/.py/;
        assert(defined($ARGV[0]), "$0 requires output filename argument");
        if(!-e catfile(curdir(), $s, $a, $ARGV[0])){
            run ['python3', $python_script, catfile(curdir(), $s, $a, $data[0])],
                '>', catfile(curdir(), $s, $a, $ARGV[0])
                or die "$python_script failed!";
        }
    }
});
