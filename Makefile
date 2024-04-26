# make this easily reproducible
#SUBJECTS = $(shell perl -Mv5.32 '-MFile::Slurp qw(read_dir)' -e 'map {say} sort grep {/^AB\d+$$/} read_dir(".")')
PERL = perl -Mv5.32 -Mutf8 -Mstrict -Mwarnings '-MFile::Slurp qw(read_dir)' -MFile::Spec::Functions
PYTHON3 = python3

# a pickle of all the data and window indices, for storage in memory during training
GUD = GrandUnifiedData.pickle
# and a normalized version, with column-wise mean and std dev as well
GUD_NORMAL = GrandUnifiedData_normalized.pickle

# hack, but we don't want to rebuild $(GUD) if it already exists,
# but the normal target will always build preprocessed_data since it's phony
default:
	[ -e $(GUD_NORMAL) ] || wget -O $(GUD_NORMAL) TODO || (rm $(GUD_NORMAL) && $(MAKE) $(GUD_NORMAL))

# ideally these would be separate targets but...
preprocessed_data: ProcessedData.zip
	# unzip data if it looks like it hasn't been
	$(PERL) -e '`unzip $^` if !grep {/^AB\d+$$/} read_dir(curdir())'
	# preprocess the data
	## downsample emg data
	$(PERL) foreach.pl emg_downsample.py emg emg_downsampled.csv
	## aggregate the sensor data for each subject and activity
	$(PERL) aggregate.pl aggregate.csv
	## filter for rows where at least one of left or right is active
	$(PERL) foreach.pl filter.py aggregate preprocessed_data.csv

$(GUD):
	# single file pickle of data and window indices
	$(MAKE) preprocessed_data
	$(PYTHON3) pickle_data.py preprocessed_data.csv $@
	ls --human-readable --size $@

$(GUD_NORMAL): $(GUD)
	# single file pickle of normalized data, window indices, and column-wise sum counts and std dev
	$(PYTHON3) normalize.py $(GUD) $@
	ls --human-readable --size $@

ProcessedData.zip:
	wget -O $@ https://repository.gatech.edu/bitstreams/03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/download

check_data.txt: $(GUD) $(GUD_NORMAL)
	$(PYTHON3) check_pickle.py > $@

clean:
	find . -maxdepth 1 -type d -name 'AB[0-9]*' -exec rm -r '{}' +
	-rm $(GUD) $(GUD_NORMAL)
.PHONY: preprocessed_data clean
