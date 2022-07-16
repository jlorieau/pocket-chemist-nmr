#!/bin/csh

#-----------------------------------------------------#
# Setting -xT for SMILE below to be the same value as #
# for -yT in fid.com prevents SMILE from extending    #
# the indirect dimension acquisition time by 50%.     #
# This is because in this TROSY 2D data the indirect  #
# dimension has been acquired for 1.017 s and the     #
# signal has already fully decayed. The default       #
# extension is intended for improving the resolution  #
# of the often truncated indirect dimensions in 3D/4D #
#-----------------------------------------------------#
# The default value for the max Time Domain Length    #
# (-maxTDL, defined as the acquisition time Tacq/T2)  #
# is 3. This means that the lineshape simulation is   #
# performed with the R2 changing from zero to 3/Tacq. #
# This is ideal for 3D or 4D for which the indirect   #
# acquisition time is typically well within 3 times   #
# T2. In this example, however, the indirect dim has  #
# been sampled for more than 3 times T2. The R2 range #
# in the lineshape simulation therefore needs to be   #
# increased. Setting -maxTDL to 5 is sufficient       #
#-----------------------------------------------------#

nmrPipe -in test.fid                                  \
| nmrPipe  -fn POLY -time                             \
| nmrPipe  -fn GMB -lb -4 -gb 0.8 -c 1.0              \
| nmrPipe  -fn ZF -zf 2 -auto                         \
| nmrPipe  -fn FT                                     \
| nmrPipe  -fn PS -p0 -24 -p1 0 -di                   \
| nmrPipe  -fn POLY -auto -ord 2 -x1 10ppm -xn 6ppm   \
| nmrPipe  -fn EXT -x1 11.5ppm -xn 6.5ppm -sw -round 2 \
| nmrPipe  -fn TP                                     \
| nmrPipe  -fn SMILE -nDim 2 -sample nuslist          \
           -maxIter 300 -nSigma 4 -nThread 4          \
           -xP0 90 -xP1 0 -xT 181 -report 2 -maxTDL 5\
| nmrPipe  -fn ZF -zf 2 -auto                         \
| nmrPipe  -fn FT                                     \
| nmrPipe  -fn PS -p0  90  -p1  0  -di                \
| nmrPipe  -fn TP                                     \
   -verb -ov -out smile.ft2

