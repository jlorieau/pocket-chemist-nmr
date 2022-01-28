#!/bin/csh

set PULPROG=`grep PULPROG\= acqus|sed 's/.*<//g'|sed 's/>//g'`


nmrPipe -in spec.fid \
#| nmrPipe -fn EM -lb 1 \
| nmrPipe  -fn SP -off 0.45 -end 0.95 -pow 1 -c 0.5 \
| nmrPipe -fn ZF -size 8192 \
| nmrPipe -fn FT -auto \
#| nmrPipe -fn EXT -x1 11.5ppm -xn -1.5ppm -sw \
| nmrPipe -fn PS -p0 0. -p1 0. \
#| nmrPipe -fn PS -p0 57.0 -p1 -36. -di \
#| nmrPipe -fn SIGN -right \
#| nmrPipe -fn CBF \
 -ov -out ${PULPROG}_complex.ft

