# make this easily reproducible
#SUBJECTS = $(shell perl -Mv5.32 '-MFile::Slurp qw(read_dir)' -e 'map {say} sort grep {/^AB\d+$$/} read_dir(".")')
PERL = perl -Mv5.32 -Mutf8 -Mstrict -Mwarnings '-MFile::Slurp qw(read_dir)' -MFile::Spec::Functions
PYTHON3 = python3

GUD = GrandUnifiedData.pickle

default: ProcessedData.zip
	# unzip data if it looks like it hasn't been
	## in the ideal case this would be a target, but...
	$(PERL) -e '`unzip $^` if !grep {/^AB\d+$$/} read_dir(curdir())'
	# preprocess the data
	## downsample emg data
	$(PERL) foreach.pl emg_downsample.py emg emg_downsampled.csv
	## aggregate the sensor data for each subject and activity
	$(PERL) aggregate.pl aggregate.csv
	## filter for rows where at least one of left or right is active
	$(PERL) foreach.pl filter.py aggregate preprocessed_data.csv

$(GUD): default
	$(PYTHON3) pickle_data.py preprocessed_data.csv $@

ProcessedData.zip:
	wget -O $@ https://repository.gatech.edu/bitstreams/03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/download

clean:
	find . -maxdepth 1 -type d -name 'AB[0-9]*' -exec rm -r '{}' +
.PHONY: default clean
