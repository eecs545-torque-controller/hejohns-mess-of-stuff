#!/usr/bin/env perl

use v5.36;
use utf8;
use strict;
use warnings;

use Carp::Assert;
use Carp;
use List::Util qw(zip);
#use List::MoreUtils;
use File::Slurp;

#use PDL::Graphics::Gnuplot;
use Text::CSV;
use JSON;
#use Hash::Merge;
use Devel::Size;

sub read_csv{
    my ($filename, $sensorfamily) = @_;
    # figure out which columns to read
    my @sensors;
    assert(defined $sensorfamily);
    if($sensorfamily eq 'activity'){
        @sensors = (
            'left',
            'right'
        );
    }
    elsif($sensorfamily eq 'moment_filt'){
        @sensors = (
            'knee_angle_l_moment',
            'ankle_angle_l_moment',
            'subtalar_angle_l_moment',
        );
    }
    elsif($sensorfamily eq 'angle'){
        @sensors = (
            'hip_flexion_l',
            'hip_adduction_l',
            'hip_rotation_l',

            'hip_flexion_r',
            'hip_adduction_r',
            'hip_rotation_r',

            'knee_angle_r',
            'ankle_angle_r',
        );
    }
    elsif($sensorfamily eq 'emg2'){
        @sensors = (
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
        );
    }
    elsif($sensorfamily eq 'imu_real'){
        @sensors = (
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
        );
    }
    elsif($sensorfamily eq 'velocity'){
        @sensors = (
            'hip_flexion_velocity_l',
            'hip_adduction_velocity_l',
            'hip_rotation_velocity_l',

            'hip_flexion_velocity_r',
            'hip_adduction_velocity_r',
            'hip_rotation_velocity_r',
            'knee_velocity_r',
            'ankle_velocity_r',
            'subtalar_velocity_r',
        );
    }
    else{
        croak "Did not expect $sensorfamily";
    }
    assert(@sensors);
    my %kv = %{Text::CSV::csv ({
            # attributes
            binary => 0,
            decode_utf8 => 0,
            strict => 1,

            # csv arguments
            in => $filename,
            encoding => 'UTF-8',
            key => 'time',
            value => \@sensors,
        }) or confess Text::CSV->error_diag};
    return \%kv;
    # Can't select column names easily
    #return rcsv1D($filename, $column_ids);
}

STDOUT->autoflush(1);

# NOTE: aggregate data here, but only use one subject and one activity for now
my @subjects = sort(grep {/AB0[12345678]/} File::Slurp::read_dir('.'));
#my @subjects = sort(grep {/AB\d+/} File::Slurp::read_dir('.'));
#@subjects = sort(grep {/AB\d+/} @subjects);
say 'Using subjects:';
say join "\n", map {"\t" . $_} @subjects;

# make all the pipes
my %pipes;
for (@subjects){
    pipe(my $read_handle, my $write_handle) or croak 'pipe failed!';
    $read_handle->blocking(1);
    $write_handle->autoflush(1);
    $pipes{$_} = [$read_handle, $write_handle];
}

# we have to aggregate the data in yet another thread to avoid deadlock
my $pid = fork;
defined $pid or croak 'fork failed!';
if($pid){ # main process
    for (@subjects){
        close $pipes{$_}[1]; # close the write handles for this process
    }
    # merge data
    my %total_data;
    for (@subjects){
        say "[debug] waiting for $_";
        my %subject_data = %{JSON::decode_json (do {
            local $/ = undef;
            my $read_fh = $pipes{$_}[0]; # this is a perl gotcha
            my $slice = <$read_fh>;
            close $read_fh;
            $slice;
        })};
        say "[debug] getting subject_data for $_";
        say "[debug] got keys ${\(keys %subject_data)}";
        #say JSON::to_json \%subject_data;
        say "[debug] total_data size before merge is ", Devel::Size::total_size(\%total_data);
        # more memory efficient merge?
        #my %total_data = %{Hash::Merge::merge(\%total_data, \%subject_data)};
        # should be largely redundant since there's only one key
        for (keys %subject_data){
            $total_data{$_} = $subject_data{$_};
            delete $subject_data{$_};
        }
        say "[debug] total_data size after merge is ", Devel::Size::total_size(\%total_data);
    }
    # TOOD: running out of memory for the JSON
    #File::Slurp::write_file('./total_data.json', JSON::to_json \%total_data);
    open(my $fh, '>', './total_data.json') or croak 'open failed!';
    print $fh '{';
    my @keys = keys %total_data;
    for (my $i = 0; $i < @keys; $i++){
        print $fh ', ' if $i;
        print $fh "\"$keys[$i]\": ";
        print $fh JSON::to_json $total_data{$keys[$i]};
        delete $total_data{$keys[$i]};
    }
    print $fh '}';
    close $fh;
}
else{ # child that spawns off the worker processes
    my @children;
    for my $s (@subjects){
        # wait if there are too many children running
        # or else it starts thrashing my machine
        while(@children > 5){
            waitpid $children[0], 0;
            shift @children;
        }
        my $pid = fork;
        defined $pid or croak 'fork failed!';
        if($pid){ # parent
            push @children, $pid;
            close $pipes{$s}[1]; # need to close all writers for EOF
            next;
        }
        else{ # child
            say "child $s beginning";
            my $read_handle = $pipes{$s}[0];
            close $read_handle;
            # collate data for this subject
            my $subject_data;
            my @activities = sort(File::Slurp::read_dir("./$s/"));
            #say "Activities for subject $s:";
            #say join "\n", map {"\t\t" . $_} @activities;

            for my $a (@activities){
                my @data = sort(grep {/(moment_filt)|(angle)|(emg2)|(imu_real)|(velocity)\.csv$/} File::Slurp::read_dir("./$s/$a"));
                #say "Data for $a for $s:";
                #say join "\n", map {"\t\t\t" . $_} @data;

                for my $d (@data){
                    $d =~ /(activity|angle|emg2|grf|imu_real|imu_sum|insole_sim|moment|moment_filt|power|velocity)\.csv$/;
                    my $sensorfamily = $1;
                    say "[debug] $d";
                    say "[debug] $sensorfamily";
                    $subject_data->{$s}{$a}{$d} = read_csv("./$s/$a/$d", $sensorfamily);
                    assert(ref $subject_data->{$s}{$a}{$d} eq 'HASH');
                }
                assert(ref $subject_data->{$s}{$a} eq 'HASH');
            }
            assert(ref $subject_data->{$s} eq 'HASH');
            assert(ref $subject_data eq 'HASH');

            # send to parent
            say "sending $s data to parent";
            my $write_handle = $pipes{$s}[1];
            print $write_handle JSON::encode_json($subject_data);
            close $write_handle;
            exit 0;
        }
    }
}

=cut

my ($time, $left, $right) = rcsv1D("AB01/normal_walk_1_0-6/AB01_normal_walk_1_0-6_activity_flag.csv");
assert($time->nelem() == $left->nelem() && $left->nelem() == $right->nelem());
my @z = zip([list $left], [list $right]);
#say (pdl @z);

my ($_time,
    $hip_flexion_r_moment,
    $hip_adduction_r_moment,
    $hip_rotation_r_moment,
    $hip_flexion_l_moment,
    $hip_adduction_l_moment,
    $hip_rotation_l_moment,
    $knee_angle_r_moment,
    $knee_angle_l_moment,
    $ankle_angle_r_moment,
    $subtalar_angle_r_moment,
    $ankle_angle_l_moment,
    $subtalar_angle_l_moment,
    ) = rcsv1D("AB01/normal_walk_1_0-6/AB01_normal_walk_1_0-6_moment_filt.csv");
assert(all $time == $_time);

my ($__time,
    $hip_flexion_r,
    $hip_adduction_r,
    $hip_rotation_r,
    $knee_angle_r,
    $ankle_angle_r,
    $subtalar_angle_r,
    $hip_flexion_l,
    $hip_adduction_l,
    $hip_rotation_l,
    $knee_angle_l,
    $ankle_angle_l,
    $subtalar_angle_l,
    ) = rcsv1D("AB01/normal_walk_1_0-6/AB01_normal_walk_1_0-6_angle.csv");
assert(all $time == $__time);

#say $knee_angle_r;

my $w = gpwin();
my $x = $time;
my $y = $knee_angle_l;
my $x2 = $time;
my $y2 = $knee_angle_r;
$w->plot(legend => 'test', $x, $y, legend => 'test2', $x2, $y2);
#$w->plot(legend => 'test2', $x2, $y2);
$w->pause_until_close();
