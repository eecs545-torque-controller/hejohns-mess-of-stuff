# make this easily reproducible
#SUBJECTS = $(shell perl -Mv5.32 '-MFile::Slurp qw(read_dir)' -e 'map {say} sort grep {/^AB\d+$$/} read_dir(".")')
PERL = perl -Mv5.32 -Mutf8 -Mstrict -Mwarnings '-MFile::Slurp qw(read_dir)' -MFile::Spec::Functions
PYTHON3 = python3

# a pickle of all the data and window indices, for storage in memory during training
GUD = GrandUnifiedData.50.pickle
# and a normalized version, with column-wise mean and std dev as well
GUD_NORMAL = GrandUnifiedData_normalized.50.pickle
GUD_NORMAL100 = GrandUnifiedData_normalized.100.pickle
GUD_NORMAL200 = GrandUnifiedData_normalized.200.pickle

# hack, but we don't want to rebuild $(GUD) if it already exists,
# but the normal target will always build preprocessed_data since it's phony
default:
	[ -e $(GUD_NORMAL) ] || wget -O $(GUD_NORMAL) https://tempestj.ddns.net/s/CLfXEDkzNRXY87C/download || (rm $(GUD_NORMAL) && $(MAKE) $(GUD_NORMAL))
	[ -e $(GUD_NORMAL100) ] || wget -O $(GUD_NORMAL100) todo || (rm $(GUD_NORMAL100) && $(MAKE) $(GUD_NORMAL100))
	[ -e $(GUD_NORMAL200) ] || wget -O $(GUD_NORMAL200) todo || (rm $(GUD_NORMAL200) && $(MAKE) $(GUD_NORMAL200))
	# this should fail if we build everything afresh
	-md5sum --status --strict -w -c checksums
	#$(MAKE) check_data.txt

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
	du -h $@

$(GUD_NORMAL): $(GUD)
	# single file pickle of normalized data, window indices, and column-wise sum counts and std dev
	$(PYTHON3) normalize.py $< $@
	du -h $@

$(GUD_NORMAL100):
	[ -e $(GUD_NORMAL) ] || $(MAKE) $(GUD_NORMAL)
	$(PYTHON3) recalc_window.py $(GUD_NORMAL) $@

$(GUD_NORMAL200):
	[ -e $(GUD_NORMAL) ] || $(MAKE) $(GUD_NORMAL)
	$(PYTHON3) recalc_window.py $(GUD_NORMAL) $@

ProcessedData.zip:
	wget -O $@ https://repository.gatech.edu/bitstreams/03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/download

check_data.txt: $(GUD) $(GUD_NORMAL)
	$(PYTHON3) check_pickle.py > $@

clean:
	find . -maxdepth 1 -type d -name 'AB[0-9]*' -exec rm -r '{}' +
	-rm $(GUD) $(GUD_NORMAL)
.PHONY: preprocessed_data clean
