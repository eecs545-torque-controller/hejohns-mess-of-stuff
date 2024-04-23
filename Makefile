# make this easily reproducible
#SUBJECTS = $(shell perl -Mv5.32 '-MFile::Slurp qw(read_dir)' -e 'map {say} sort grep {/^AB\d+$$/} read_dir(".")')
PERL = perl -Mv5.32 '-MFile::Slurp qw(read_dir)'

default: ProcessedData.zip
	# unzip data if it looks like it hasn't been
	$(PERL) -e '`unzip $^` if !grep {/^AB\d+$$/} read_dir(".")'

ProcessedData.zip:
	wget -O $@ https://repository.gatech.edu/bitstreams/03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/download

clean:
	-find . -type d -name 'AB[0-9]*' -delete
.PHONY: default clean
