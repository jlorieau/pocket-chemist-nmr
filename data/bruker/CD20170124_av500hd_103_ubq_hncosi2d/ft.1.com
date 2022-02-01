#!/bin/csh

#
# Processing of a 2D Plane from 3D States HN-Detected data:
#   "POLY -time" subtracts solvent by polynomial fitting.
#   "EXT -left" extracts left half of HN dimension.
 
rm -rf ft/

xyz2pipe -in fid/spec%03d.fid -x -verb \
#| nmrPipe  -fn SOL \
#| nmrPipe  -fn POLY -time                             \
#| nmrPipe  -fn SP -off 0.5 -end 0.98 -pow 1 -c 1.0    \
#| nmrPipe  -fn ZF -size 2048 \
| nmrPipe  -fn FT             \
#| nmrPipe  -fn EXT -x1 11.5ppm -xn 6.0ppm -sw    \
#| nmrPipe  -fn EXT -x1 11.5ppm -xn 6.5ppm -sw    \
#| nmrPipe  -fn PS -p0 0.  -p1 0.0 -di  \
#| nmrPipe  -fn PS -p0 -30.8  -p1 -25.0 -di  \
| pipe2xyz -out ft/spec%03d.ft1 -x
