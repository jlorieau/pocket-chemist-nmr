# Build test datasets

all: spec.fid hsqcetfpf3gpsi2.ft2 hsqcetfpf3gpsi2_complex.ft2

spec.fid: ser
	./fid.com

hsqcetfpf3gpsi2.ft2: spec.fid
	./ft12.com

hsqcetfpf3gpsi2_complex.ft2: spec.fid
	./ft12_complex.com
