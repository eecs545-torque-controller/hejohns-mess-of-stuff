# make this easily reproducible
#SUBJECTS = $(shell perl -Mv5.32 '-MFile::Slurp qw(read_dir)' -e 'map {say} sort grep {/^AB\d+$$/} read_dir(".")')
PERL = perl -Mv5.32 -Mutf8 -Mstrict -Mwarnings '-MFile::Slurp qw(read_dir)'
PYTHON3 = python3

default: ProcessedData.zip
	# unzip data if it looks like it hasn't been
	## in the ideal case this would be a target, but...
	$(PERL) -e '`unzip $^` if !grep {/^AB\d+$$/} read_dir(".")'
	# preprocess the data
	## aggregate the sensor data for each subject and activity
	$(PERL) aggregate.pl aggregate.csv

ProcessedData.zip:
	wget -O $@ https://repository.gatech.edu/bitstreams/03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/download

clean:
	-find . -type d -name 'AB[0-9]*' -delete
.PHONY: default clean
