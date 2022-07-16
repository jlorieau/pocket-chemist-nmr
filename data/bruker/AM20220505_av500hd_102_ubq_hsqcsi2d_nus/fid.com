#!/bin/csh

nusExpand.tcl -mode bruker -sampleCount 36 -off 0 \
 -in ./ser -out ./ser_full -sample ./nuslist

bruk2pipe -verb -in ./ser_full \
  -bad 0.0 -ext -aswap -AMX -decim 3328 -dspfvs 20 -grpdly 67.9842681884766  \
  -xN              1024  -yN               362  \
  -xT               399  -yT               181  \
  -xMODE            DQD  -yMODE  Echo-AntiEcho  \
  -xSW         6009.615  -ySW         1773.050  \
  -xOBS         499.872  -yOBS          50.657  \
  -xCAR           4.773  -yCAR         117.559  \
  -xLAB              HN  -yLAB             15N  \
  -ndim               2  -aq2D         Complex  \
| nmrPipe -fn MULT -c 9.76562e-01 \
  -out ./test.fid -ov

sleep 5
