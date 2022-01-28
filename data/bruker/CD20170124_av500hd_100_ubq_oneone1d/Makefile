# Build test datasets

all: spec.fid oneone-echo_N-dcpl.jll.ft oneone-echo_N-dcpl.jll_complex.ft

spec.fid: fid
	./fid.com

oneone-echo_N-dcpl.jll.ft: spec.fid
	./ft.com

oneone-echo_N-dcpl.jll_complex.ft: spec.fid
	./ft_complex.com
