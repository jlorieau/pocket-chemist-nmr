"""
Test spectrum utilities
"""
import torch

from pocketchemist_nmr.spectra.utils import (combine_block_from_complex,
                                             combine_single_from_complex,
                                             split_block_to_complex,
                                             split_single_to_complex,
                                             interleave_block_to_single,
                                             interleave_single_to_block)


def test_interleave_block_to_single():
    """Test interleave_block_to_single function"""
    # Create the test data
    # Block interleave along N, single interleave along M
    N = 3  # Number of complex points on X-axis
    M = 5  # Number of complex points on Y-axis

    #  [[ 0.,  1.,  2.,  3.,  4.,  5.],  (Real XN + Imag XN / Real Y1)
    #   [ 6.,  7.,  8.,  9., 10., 11.],  (Real XN + Imag XN / Imag Y1)
    #   [12., 13., 14., 15., 16., 17.],  (Real XN + Imag XN / Real Y2)
    #   [18., 19., 20., 21., 22., 23.],  (Real XN + Imag XN / Imag Y2)
    #   [24., 25., 26., 27., 28., 29.],  (Real XN + Imag XN / Real Y3)
    #   [30., 31., 32., 33., 34., 35.],  (Real XN + Imag XN / Real Y3)
    #   [36., 37., 38., 39., 40., 41.],  (Real XN + Imag XN / Real Y4)
    #   [42., 43., 44., 45., 46., 47.],  (Real XN + Imag XN / Real Y4)
    #   [48., 49., 50., 51., 52., 53.],  (Real XN + Imag XN / Real Y5)
    #   [54., 55., 56., 57., 58., 59.]]  (Real XN + Imag XN / Real Y5)
    data = torch.arange(float(N * 2 * M * 2)).reshape(M * 2, N * 2)

    # Single interleave the dataset
    # [[ 0.,  3.,  1.,  4.,  2.,  5.],
    #  [ 6.,  9.,  7., 10.,  8., 11.],
    #  [12., 15., 13., 16., 14., 17.],
    #  [18., 21., 19., 22., 20., 23.],
    #  [24., 27., 25., 28., 26., 29.],
    #  [30., 33., 31., 34., 32., 35.],
    #  [36., 39., 37., 40., 38., 41.],
    #  [42., 45., 43., 46., 44., 47.],
    #  [48., 51., 49., 52., 50., 53.],
    #  [54., 57., 55., 58., 56., 59.]]
    single_interleave = interleave_block_to_single(data)

    # Check the size and the first and last row
    assert single_interleave.size() == (M * 2, N * 2)
    assert tuple(single_interleave[0]) == (0., 3., 1., 4., 2., 5.)
    assert tuple(single_interleave[-1]) == (54., 57., 55., 58., 56., 59.)

    # The tensors use the same storage
    #assert id(data.storage()) == id(single_interleave.storage())


def test_interleave_single_to_block():
    """Test interleave_single_to_block function."""
    # Create the test data
    # Block interleave along N, single interleave along M
    N = 3  # Number of complex points on X-axis
    M = 5  # Number of complex points on Y-axis

    #  [[ 0.,  1.,  2.,  3.,  4.,  5.],  (Real XN + Imag XN / Real Y1)
    #   [ 6.,  7.,  8.,  9., 10., 11.],  (Real XN + Imag XN / Imag Y1)
    #   [12., 13., 14., 15., 16., 17.],  (Real XN + Imag XN / Real Y2)
    #   [18., 19., 20., 21., 22., 23.],  (Real XN + Imag XN / Imag Y2)
    #   [24., 25., 26., 27., 28., 29.],  (Real XN + Imag XN / Real Y3)
    #   [30., 31., 32., 33., 34., 35.],  (Real XN + Imag XN / Real Y3)
    #   [36., 37., 38., 39., 40., 41.],  (Real XN + Imag XN / Real Y4)
    #   [42., 43., 44., 45., 46., 47.],  (Real XN + Imag XN / Real Y4)
    #   [48., 49., 50., 51., 52., 53.],  (Real XN + Imag XN / Real Y5)
    #   [54., 55., 56., 57., 58., 59.]]  (Real XN + Imag XN / Real Y5)
    data = torch.arange(float(N * 2 * M * 2)).reshape(M * 2, N * 2)

    # Block interleave the dataset
    # [[ 0.,  2.,  4.,  1.,  3.,  5.],
    #  [ 6.,  8., 10.,  7.,  9., 11.],
    #  [12., 14., 16., 13., 15., 17.],
    #  [18., 20., 22., 19., 21., 23.],
    #  [24., 26., 28., 25., 27., 29.],
    #  [30., 32., 34., 31., 33., 35.],
    #  [36., 38., 40., 37., 39., 41.],
    #  [42., 44., 46., 43., 45., 47.],
    #  [48., 50., 52., 49., 51., 53.],
    #  [54., 56., 58., 55., 57., 59.]]
    block_interleave = interleave_single_to_block(data)

    # Check the size and the first and last row
    assert block_interleave.size() == (M * 2, N * 2)
    assert tuple(block_interleave[0]) == (0., 2., 4., 1., 3., 5.)
    assert tuple(block_interleave[-1]) == (54., 56., 58., 55., 57., 59.)

    # The tensors use the same storage
    #assert id(data.storage()) == id(block_interleave.storage())


def test_combine_split_block_complex_hypercomplex_2d():
    """Test the combine_block_from_complex and split_block_to_complex functions
    with a hypercomplex 2D"""
    # Create a test 2D (from fdatap.h)
    # (N X-Axis=Real Values for Y-Axis Increment 1 Real)
    # (N X-Axis=Imag Values for Y-Axis Increment 1 Real)
    # (N X-Axis=Real Values for Y-Axis Increment 1 Imag)
    # (N X-Axis=Imag Values for Y-Axis Increment 1 Imag)
    # ...
    # (N X-Axis=Real Values for Y-Axis Increment M Imag)
    # (N X-Axis=Imag Values for Y-Axis Increment M Imag)
    # Block interleave along N, single interleave along M
    N = 3  # Number of complex points on X-axis
    M = 5  # Number of complex points on Y-axis

    #  [[ 0.,  1.,  2.,  3.,  4.,  5.],  (Real XN + Imag XN / Real Y1)
    #   [ 6.,  7.,  8.,  9., 10., 11.],  (Real XN + Imag XN / Imag Y1)
    #   [12., 13., 14., 15., 16., 17.],  (Real XN + Imag XN / Real Y2)
    #   [18., 19., 20., 21., 22., 23.],  (Real XN + Imag XN / Imag Y2)
    #   [24., 25., 26., 27., 28., 29.],  (Real XN + Imag XN / Real Y3)
    #   [30., 31., 32., 33., 34., 35.],  (Real XN + Imag XN / Real Y3)
    #   [36., 37., 38., 39., 40., 41.],  (Real XN + Imag XN / Real Y4)
    #   [42., 43., 44., 45., 46., 47.],  (Real XN + Imag XN / Real Y4)
    #   [48., 49., 50., 51., 52., 53.],  (Real XN + Imag XN / Real Y5)
    #   [54., 55., 56., 57., 58., 59.]]  (Real XN + Imag XN / Real Y5)
    data = torch.arange(float(N * 2 * M * 2)).reshape(M * 2, N * 2)

    # Split to complex into a real/complex dataset
    # [[ 0.+3.j,  1.+4.j,  2.+5.j],
    #  [ 6.+9.j,  7.+10.j,  8.+11.j],
    #  [12.+15.j, 13.+16.j, 14.+17.j],
    #  [18.+21.j, 19.+22.j, 20.+23.j],
    #  [24.+27.j, 25.+28.j, 26.+29.j],
    #  [30.+33.j, 31.+34.j, 32.+35.j],
    #  [36.+39.j, 37.+40.j, 38.+41.j],
    #  [42.+45.j, 43.+46.j, 44.+47.j],
    #  [48.+51.j, 49.+52.j, 50.+53.j],
    #  [54.+57.j, 55.+58.j, 56.+59.j]]
    cmplx = split_block_to_complex(data)

    # Check the size and the first and last row
    assert cmplx.size() == (M * 2, N)
    assert tuple(cmplx[0]) == (0. + 3.j, 1. + 4.j, 2 + 5.j)
    assert tuple(cmplx[-1]) == (54. + 57.j, 55. + 58.j, 56 + 59.j)

    # Reorganize into a real/real tensor
    combined = combine_block_from_complex(cmplx)
    assert torch.all(torch.eq(combined, data))


def test_combine_split_single_complex_hypercomplex_2d():
    """Test the combine_single_from_complex and split_single_to_complex
    functions with a hypercomplex 2D"""
    # Create a test 2D (from fdatap.h)
    # (N X-Axis=Real Values for Y-Axis Increment 1 Real)
    # (N X-Axis=Imag Values for Y-Axis Increment 1 Real)
    # (N X-Axis=Real Values for Y-Axis Increment 1 Imag)
    # (N X-Axis=Imag Values for Y-Axis Increment 1 Imag)
    # ...
    # (N X-Axis=Real Values for Y-Axis Increment M Imag)
    # (N X-Axis=Imag Values for Y-Axis Increment M Imag)
    # Block interleave along N, single interleave along M
    N = 3  # Number of complex points on X-axis
    M = 5  # Number of complex points on Y-axis

    #  [[ 0.,  1.,  2.,  3.,  4.,  5.],  (Real XN + Imag XN / Real Y1)
    #   [ 6.,  7.,  8.,  9., 10., 11.],  (Real XN + Imag XN / Imag Y1)
    #   [12., 13., 14., 15., 16., 17.],  (Real XN + Imag XN / Real Y2)
    #   [18., 19., 20., 21., 22., 23.],  (Real XN + Imag XN / Imag Y2)
    #   [24., 25., 26., 27., 28., 29.],  (Real XN + Imag XN / Real Y3)
    #   [30., 31., 32., 33., 34., 35.],  (Real XN + Imag XN / Real Y3)
    #   [36., 37., 38., 39., 40., 41.],  (Real XN + Imag XN / Real Y4)
    #   [42., 43., 44., 45., 46., 47.],  (Real XN + Imag XN / Real Y4)
    #   [48., 49., 50., 51., 52., 53.],  (Real XN + Imag XN / Real Y5)
    #   [54., 55., 56., 57., 58., 59.]]  (Real XN + Imag XN / Real Y5)
    data = torch.arange(float(N * 2 * M * 2)).reshape(M * 2, N * 2)

    # Split to complex into a real/complex dataset
    # [[ 0.+1.j,  2.+3.j,  4.+5.j],
    #  [ 6.+7.j,  8.+9.j, 10.+11.j],
    #  [12.+13.j, 14.+15.j, 16.+17.j],
    #  [18.+19.j, 20.+21.j, 22.+23.j],
    #  [24.+25.j, 26.+27.j, 28.+29.j],
    #  [30.+31.j, 32.+33.j, 34.+35.j],
    #  [36.+37.j, 38.+39.j, 40.+41.j],
    #  [42.+43.j, 44.+45.j, 46.+47.j],
    #  [48.+49.j, 50.+51.j, 52.+53.j],
    #  [54.+55.j, 56.+57.j, 58.+59.j]]
    cmplx = split_single_to_complex(data)

    # Check the size and the first and last row
    assert cmplx.size() == (M * 2, N)
    assert tuple(cmplx[0]) == (0. + 1.j, 2. + 3.j, 4. + 5.j)
    assert tuple(cmplx[-1]) == (54. + 55.j, 56. + 57.j, 58 + 59.j)

    # Reorganize into a real/real tensor
    combined = combine_single_from_complex(cmplx)
    assert torch.all(torch.eq(combined, data))

