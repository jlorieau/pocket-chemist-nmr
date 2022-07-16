#!/bin/csh

set PULPROG=`grep PULPROG\= acqus|sed 's/.*<//g'|sed 's/>//g'`


nmrPipe -in test.fid \
| nmrPipe -fn SOL \
| nmrPipe  -fn SP -off 0.45 -end 0.95 -pow 1 -c 0.5 \
| nmrPipe -fn ZF -size 4096 \
| nmrPipe -fn FT -auto \
| nmrPipe -fn EXT -x1 11.5ppm -xn 6.5ppm -sw \
| nmrPipe -fn PS -p0 0. -p1 0. -di \
#| nmrPipe -fn PS -p0 57.0 -p1 -36. -di \
| nmrPipe -fn TP \
| nmrPipe  -fn SP -off 0.45 -end 0.95 -pow 1 -c 0.5 \
| nmrPipe -fn ZF \
| nmrPipe -fn FT -auto \
| nmrPipe -fn PS -p0 -90. -p1 0. -di \
#| nmrPipe -fn PS -p0 0. -p1 0. -di \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto \
| nmrPipe -fn TP \
 -ov -out $PULPROG.ft2


