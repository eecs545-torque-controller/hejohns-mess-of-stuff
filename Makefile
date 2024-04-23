# make this easily reproducible
#SUBJECTS = $(shell perl -Mv5.32 '-MFile::Slurp qw(read_dir)' -e 'map {say} sort grep {/^AB\d+$$/} read_dir(".")')
PERL = perl -Mv5.32 '-MFile::Slurp qw(read_dir)'

default: ProcessedData.zip
	$(PERL) -e '`unzip $^` if !grep {/^AB\d+$$/} read_dir(".")'
ProcessedData.zip:
	wget -O $@ https://repository.gatech.edu/bitstreams/03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/download
clean:
	-perl -Mv5.32 ''  | rm -r AB*
.PHONY: default clean
