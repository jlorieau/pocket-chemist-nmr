#!/bin/csh

set PULPROG=`grep PULPROG\= acqus|sed 's/.*<//g'|sed 's/>//g'`


nmrPipe -in spec.fid \
| nmrPipe -fn SOL \
| nmrPipe  -fn SP -off 0.45 -end 0.95 -pow 1 -c 0.5 \
| nmrPipe -fn ZF -size 4096 \
| nmrPipe -fn FT -auto \
| nmrPipe -fn EXT -x1 10.25ppm -xn 6.25ppm -sw \
#| nmrPipe -fn PS -p0 0. -p1 0. -di \
| nmrPipe -fn PS -p0 145.0 -p1 0. \
| nmrPipe -fn TP \
| nmrPipe  -fn SP -off 0.45 -end 0.95 -pow 1 -c 0.5 \
| nmrPipe -fn ZF \
| nmrPipe -fn FT -auto \
| nmrPipe -fn PS -p0 -90. -p1 0. \
#| nmrPipe -fn PS -p0 0. -p1 0. -di \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto \
#| nmrPipe -fn TP \
 -ov -out ${PULPROG}_complex.ft2


