The following table describes metadata for the test datasets in this
directory.

| Filename                     | ORD  | NPTS   | FDSIZE | FDSPEC | FDSLICECOUNT0 | Inner Loop |
|------------------------------|------|--------|--------|--------|---------------|------------|
| spec.fid                     | 2, 1 | 640x2  | 640    | 368    | 0             | Block      |
|                              |      | 184x2  |        |        |               | Interleave |
| spec_tp.fid                  | 1, 2 | 184x2  | 184    | 1280   | 1280          | Single     |
|                              |      | 640x2  |        |        |               | Interleave |
| hsqcetfpf3gpsi2.ft2          | 1, 2 | 368x1  | 368    | 1024   | 1024          | Single     |
|                              |      | 1024x1 |        |        |               | Interleave |
| hsqcetfpf3gpsi2_complex.ft2  | 2, 1 | 1024x2 | 1024   | 736    | 736           | Single     |
|                              |      | 368x2  |        |        |               | Interleave |

ORD: Order of data according to FDDIMORDER1, FDDIMORDER2, etc.
NPTS: Number of Complex or Real points in each dimension
FDSIZE: Number of points in current dimension (Real or Complex)
FDSPEC: Total number of 1D spectrum (Real + Imag)