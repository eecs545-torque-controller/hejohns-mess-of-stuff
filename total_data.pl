#!/usr/bin/env perl

use v5.36;
use utf8;
use strict;
use warnings;

use File::Slurp;
use Carp::Assert;
use JSON;
use Text::CSV;
use Parallel::Loops;

my $parallel = Parallel::Loops->new(8);
my %total_data;
$parallel->share(\%total_data);

my @subjects = sort(grep {/AB\d+/} File::Slurp::read_dir('.'));
$parallel->foreach(\@subjects, sub {
        my $s = $_;
        my %subject_data;
        my @activities = sort(File::Slurp::read_dir("./$s/"));
        for my $a (@activities){
            my @filtered = sort(grep {/filtered\.csv$/} File::Slurp::read_dir("./$s/$a"));
            assert(@filtered == 1);
            open(my $fh, '<', "./$s/$a/$filtered[0]") or die "open failed!";
            my $csv = Text::CSV->new({
                    binary => 0,
                    decode_utf8 => 0,
                    strict => 1,
                });
            my @header = $csv->header($fh);
            my @sensors = grep {!/time/} @header;
            my %activity_data = %{Text::CSV::csv ({
                    # attributes
                    binary => 0,
                    decode_utf8 => 0,
                    strict => 1,
                    # csv arguments
                    in => "./$s/$a/$filtered[0]",
                    encoding => 'UTF-8',
                    key => 'time',
                    value => \@sensors,
                }) or die Text::CSV->error_diag};
            $subject_data{$a} = \%activity_data;
        }
        $total_data{$s} = JSON::to_json(\%subject_data, {canonical => 1});
    });
print '{';
my $i = 0;
for (@subjects){
    print ', ' if $i;
    $i++;
    print "\"$_\": ";
    print $total_data{$_};
    delete $total_data{$_};
}
print '}';
