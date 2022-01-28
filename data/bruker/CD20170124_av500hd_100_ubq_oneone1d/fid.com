#!/bin/csh

bruk2pipe -verb -in ./fid \
  -bad 0.0 -ext -aswap -AMX -decim 2000 -dspfvs 20 -grpdly 67.9862518310547  \
  -xN              1792  \
  -xT               799  \
  -xMODE            DQD  \
  -xSW        10000.000  \
  -xOBS         499.872  \
  -xCAR           4.773  \
  -xLAB              1H  \
  -ndim               1  \
| nmrPipe -fn MULT -c 2.44141e-01 \
  -out ./spec.fid -ov

sleep 5
