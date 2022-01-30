The following table describes metadata for the test datasets in this
directory.

| Dataset          | Parameter  | Value               |
|------------------|------------|---------------------|
| fid/spec001.fid  | ORD        | 2, 1, 3, 4          |
|                  | NPTS       | 559x2, 39x2, 51x2   |
|                  | FDSIZE     | 559                 |
|                  | FDSPECNUM  | 78                  |
|                  | FDF3SIZE   | 102                 |
|                  | FDF4SIZE   | 1                   |
|                  | FILECOUNT  | 102                 |
| hncogp3d.ft3     | ORD        | 2, 3, 1, 4          |
|                  | NPTS       | 805x1, 512x1, 256x1 |
|                  | FDSIZE     | 805                 |
|                  | FDSPECNUM  | 512                 |
|                  | FDF3SIZE   | 256                 |
|                  | FDF4SIZE   | 1                   |
|                  | FILECOUNT  | 1                   |

ORD: Order of data according to FDDIMORDER1, FDDIMORDER2, etc.
NPTS: Number of Complex or Real points in each dimension
FDSIZE: Number of points in current dimension (Real or Complex)
FDSPECNUM: Total number of 2D spectra (Real + Imag)
FDF3SIZE/FDF4SIZE: Total number of 3d/4d spectra (Real + Imag)