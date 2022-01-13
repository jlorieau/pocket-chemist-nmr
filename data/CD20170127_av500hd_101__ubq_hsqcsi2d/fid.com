#!/bin/csh

bruk2pipe -verb -in ./ser \
  -bad 0.0 -ext -aswap -AMX -decim 2496 -dspfvs 20 -grpdly 67.9842376708984  \
  -xN              1280  -yN               368  \
  -xT               640  -yT               184  \
  -xMODE            DQD  -yMODE  Echo-AntiEcho  \
  -xSW         8012.821  -ySW         1671.682  \
  -xOBS         499.872  -yOBS          50.657  \
  -xCAR           4.773  -yCAR         117.566  \
  -xLAB              HN  -yLAB             15N  \
  -ndim               2  -aq2D         Complex  \
| nmrPipe -fn MULT -c 1.95312e+00 \
  -out ./test.fid -ov

sleep 5
