# Build test datasets

all: fid/spec001.fid ft/spec001.ft1 ft2/spec001.ft2 ft3/spec0001.ft3

fid/spec001.fid: ser
	./fid.com

ft/spec001.ft1: fid/spec001.fid
	./ft.1.com

ft2/spec001.ft2: ft/spec001.ft1
	./ft.2.com

ft3/spec0001.ft3: ft2/spec001.ft2
	./ft.3.com
