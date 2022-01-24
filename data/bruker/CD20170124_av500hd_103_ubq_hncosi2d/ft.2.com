#!/bin/csh

#
# Processing of a 2D Plane from 3D States HN-Detected data:
#   "POLY -time" subtracts solvent by polynomial fitting.
#   "EXT -left" extracts left half of HN dimension.
 
rm -rf ft2/

xyz2pipe -in ft/test%03d.ft1 -y -verb \
#| nmrPipe -fn LP -after -pred 64 \
| nmrPipe  -fn SP -off 0.5 -end 0.90 -pow 1 -c 0.5   \
| nmrPipe  -fn ZF -size 256 \
| nmrPipe  -fn FT \
#| nmrPipe -fn EXT -x1 12.3ppm -xn -0.6ppm -sw \
| nmrPipe  -fn PS -p0 -90.  -p1 0. -di \
#| nmrPipe  -fn PS -p0 0.  -p1 0. -di \
| pipe2xyz -out ft2/test%03d.ft2 -y
