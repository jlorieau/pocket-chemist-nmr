The following is a listing of software implementation details that were 
implemented to match other software, but that I believe are implemented
incorrectly (bugs).

1. *Group delay and rolling datasets*. To account for the low pass digital 
   filter from Bruker FIDs, the current implementation rolls the dataset by
   an integer number of the group delay (floor(group_delay)) then applies 
   first order phase correction for the remainder of the FID. A better 
   approach would be to reflect the end of the FID by the amount of the group 
   delay shift. With adequate/significant apodization, the better approach
   is likely indistinguishable from the current implementation.

2. *Group delay and apodization*. NMRPipe does not appear to account for the
   group delay when apodizing FIDs. The group delay appears to be corrected only
   during Fourier transformation. A better approach would be to apply the
   group delay in the apodization and to use a time reversed apodization 
   function. For example, with exponential apodization and a group delay (grp):
   g(t) = f(t) * exp(-abs(lb * (i - grp) * dw)). With adequate/significant 
   apodization, the better approach is likely indistinguishable from the 
   current implementation.

3. *Spectral widths*. NMRPipe (FDNORIG and FDNSW) and Topspin calculate
   the position of frequencies using "FDORIG + SW*(1.0 - 1/npts)" for the first
   point and "FDORIG" for the last point. The spectral width for this data then
   becomes "FDORIG - FDORIG + SW(1.0 - 1/npts)" or "SW(1.0 -1/npts)", which is
   not correct. The first point frequency should be "FDORIG + 1.0 * sw". As a 
   consequence of the current implementation, the corresponding dwell time is 
   "(SW(1.0 - 1/npts))**-1" which is not correct. With significant 
   zero-filling, the current approach is likely indistinguishable from the 
   better approach--i.e. when zero filled to 512 points, the spectral width
   is only incorrect by 0.2% and frequencies in between are only incorrect by
   a fraction of that amount.
