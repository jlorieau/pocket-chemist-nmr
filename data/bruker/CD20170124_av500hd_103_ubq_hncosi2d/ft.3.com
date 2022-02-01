#!/bin/csh

#
# Processing of a 2D Plane from 3D States HN-Detected data:
#   "POLY -time" subtracts solvent by polynomial fitting.
#   "EXT -left" extracts left half of HN dimension.

set PULPROG=`grep PULPROG\= acqus|sed 's/.*<//g'|sed 's/>//g'`
set NUMPOINTS = `grep '\#\#\$TD=' acqu3s|awk '{print int($2/2)}'`
set TOTALNUMPOINTS = `grep '\#\#\$TD=' acqu3|awk '{print int($2/2)}'`
rm -rf ft3/ ft3T/

# REMOVE EXT!!

xyz2pipe -in ft2/spec%03d.ft2 -z -verb \
| nmrPipe  -fn EXT -xn $NUMPOINTS -sw \
#| nmrPipe  -fn LP -after  -fb -ord 32 \
| nmrPipe  -fn SP -off 0.45 -end 0.98 -pow 1 -c 0.5 \
| nmrPipe  -fn ZF -size 512 \
| nmrPipe  -fn FT -alt \
| nmrPipe  -fn PS -p0 0.0  -p1 0.0   -di \
#| nmrPipe  -fn PS -p0 90.0  -p1 180.0   -di \
#| nmrPipe  -fn POLY -auto -ord 2 \
#| nmrPipe -fn REV \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto \
| nmrPipe -fn TP \
| pipe2xyz -out ft3/spec%04d.ft3 -y

proj3D.tcl -in ft3/spec%04d.ft3

echo "$NUMPOINTS of $TOTALNUMPOINTS points collected in the third dimension"

#xyz2pipe -in ft3/test%04d.ft3 -y -verb \
#| pipe2xyz -out ft3T/test%04d.ft3 -z
#rm -rf ft3/

xyz2pipe -in ft3/spec%04d.ft3 -out ./$PULPROG.ft3

