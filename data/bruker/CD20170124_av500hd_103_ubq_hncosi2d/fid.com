#!/bin/csh

bruk2pipe -verb -in ./ser \
  -bad 0.0 -ext -aswap -AMX -decim 2858.66666666667 -dspfvs 20 -grpdly 67.9858245849609  \
  -xN              1280  -yN                78  -zN               102  \
  -xT               559  -yT                39  -zT                51  \
  -xMODE            DQD  -yMODE  Echo-AntiEcho  -zMODE        Complex  \
  -xSW         6996.269  -ySW         1671.682  -zSW         1445.922  \
  -xOBS         499.872  -yOBS          50.657  -zOBS         125.714  \
  -xCAR           4.773  -yCAR         117.565  -zCAR         176.943  \
  -xLAB              HN  -yLAB             15N  -zLAB             13C  \
  -ndim               3  -aq2D         Complex                         \
| nmrPipe -fn MULT -c 9.76562e-01 \
| pipe2xyz -x -out ./fid/spec%03d.fid -ov

sleep 5
