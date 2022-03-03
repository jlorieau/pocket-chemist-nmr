"""
Utility functions for spectra

Data can be stored in block interleave and single interleave (interleave of 1).

The following example is block interleaved on the X-axis (2N) and single
interleaved on the Y-axis. The inner loop is the last dimension, so this
dataset is (Y: single, X: block) interleaved

    N X-Real + N X-Imag / 1 Y-Real
    N X-Real + N X-Imag / 1 Y-Imag
    ...
    N X-Real + N X-Imag / M Y-Real
    N X-Real + N X-Imag / M Y-Imag

A simple transpose would produce (X: block, Y: single):

    1 Y-Real, 1 Y-Imag, ... M Y-Real, M Y-Imag / 1 X-Real
    1 Y-Real, 1 Y-Imag, ... M Y-Real, M Y-Imag / 2 X-Real
    ...
    1 Y-Real, 1 Y-Imag, ... M Y-Real, M Y-Imag / 1 Y-Imag
    1 Y-Real, 1 Y-Imag, ... M Y-Real, M Y-Imag / 2 Y-Imag
    ...
    1 Y-Real, 1 Y-Imag, ... M Y-Real, M Y-Imag / N Y-Imag
    1 Y-Real, 1 Y-Imag, ... M Y-Real, M Y-Imag / N Y-Imag

"""
import typing as t
from math import floor

import torch
from .constants import RangeType

__all__ = ('interleave_block_to_single', 'interleave_single_to_block',
           'split_block_to_complex', 'split_single_to_complex',
           'combine_block_from_complex', 'combine_single_from_complex',
           'range_endpoints', 'gen_range')


def interleave_block_to_single(tensor: torch.Tensor) -> torch.Tensor:
    """Change interleave of the last tensor dimension from block interleave
    to single interleave.

    Parameters
    ----------
    tensor
        A real tensor with block-interleaved data in the last dimension

    Returns
    -------
    tensor
        A real tensor with single-interleaved data in the last dimension
    """
    # Single Interleave the blocks
    return torch.stack(tensor.split(int(tensor.size()[-1] / 2),
                                    dim=tensor.dim() - 1),
                       dim=tensor.dim()).view(tensor.size())


def interleave_single_to_block(tensor: torch.Tensor) -> torch.Tensor:
    """Change interleave of the last tensor dimension from single interleave
    to block interleave

    Parameters
    ----------
    tensor
        A real tensor with single-interleaved data in the last dimension

    Returns
    -------
    tensor
        A real tensor with block-interleaved data in the last dimension
    """
    # Separate single interleave and interleave the blocks
    return torch.hstack((tensor[..., ::2], tensor[..., 1::2]))


def split_block_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Split a tensor with block interleaved real/imag data in the last
    dimension to a complex tensor.

    Parameters
    ----------
    tensor
        Tensor with real block-interleaved data in the last dimension

    Returns
    -------
    complex_tensor
        A complex tensor constructed from deinterleaved data in the last
        dimension.

    Examples
    --------
    >>> t = torch.arange(16.0).reshape(4, 2, 2)
    >>> t
    tensor([[[ 0.,  1.],
             [ 2.,  3.]],
    <BLANKLINE>
            [[ 4.,  5.],
             [ 6.,  7.]],
    <BLANKLINE>
            [[ 8.,  9.],
             [10., 11.]],
    <BLANKLINE>
            [[12., 13.],
             [14., 15.]]])
    >>> t.size()
    torch.Size([4, 2, 2])
    >>> cmplx = split_to_complex(t)
    >>> cmplx
    tensor([[[ 0.+2.j,  1.+3.j]],
    <BLANKLINE>
            [[ 4.+6.j,  5.+7.j]],
    <BLANKLINE>
            [[ 8.+10.j,  9.+11.j]],
    <BLANKLINE>
            [[12.+14.j, 13.+15.j]]])
    >>> cmplx.size()
    torch.Size([4, 1, 2])
    """
    return torch.complex(*torch.split(tensor, int(tensor.size()[-1] / 2),
                                      dim=tensor.dim() - 1))


def split_single_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Split a tensor with single interleaved real/imag data in the last
    dimension to a complex tensor.

    Parameters
    ----------
    tensor
        Tensor with real single-interleaved data in the last dimension

    Returns
    -------
    complex_tensor
        A complex tensor constructed from deinterleaved data in the last
        dimension.
    """
    # Separate the interleaved data into real and complex components
    return torch.complex(real=tensor[..., ::2], imag=tensor[..., 1::2])


def combine_block_from_complex(complex_tensor: torch.Tensor) -> torch.Tensor:
    """Combine a complex tensor into a real tensor with real/imag block
    interleave in the last dimension.

    Parameters
    ----------
    complex_tensor
        Tensor with complex data in the last dimension

    Returns
    -------
    tensor
        A real tensor with the real/imag components block-interleaved data in
        the last dimension.

    Examples
    --------
    >>> t1 = torch.arange(16.0).reshape(4, 2, 2)
    >>> t1.size()
    torch.Size([4, 2, 2])
    >>> t2 = combine_from_complex(split_to_complex(t1))
    >>> torch.all(torch.eq(t1, t2))  # The tensors are the same
    tensor(True)
    """
    return torch.hstack((complex_tensor.real, complex_tensor.imag))


def combine_single_from_complex(complex_tensor: torch.Tensor) -> torch.Tensor:
    """Combine a complex tensor into a real tensor with real/imag single
    interleave in the last dimension.

    Parameters
    ----------
    complex_tensor
        Tensor with complex data in the last dimension

    Returns
    -------
    tensor
        A real tensor with the real/imag components block-interleaved data in
        the last dimension.
    """
    # Single Interleave the blocks
    dim = complex_tensor.dim()
    # Increase the last dimension's size by a factor of 2
    size = tuple(s if i < dim - 1 else int(s * 2.)
                 for i, s in enumerate(complex_tensor.size()))
    return torch.stack((complex_tensor.real, complex_tensor.imag),
                       dim=dim).view(size)


def range_endpoints(npts: int,
                    range_type: RangeType = RangeType.UNIT,
                    sw: t.Optional[float] = None,
                    group_delay: t.Optional[float] = None) \
        -> t.Tuple[float, float]:
    """The endpoints for ranges, based on spectral parameters

    Parameters
    ----------
    npts
        The number of points for the range
    range_type
        The type of range to generate
    sw
        The spectral width (in Hz) for the spectrum
    group_delay
        The optional group delay (in pts) to apply in offsetting time ranges

    Returns
    -------
    start, end
        A start and end points of the range values
    """
    # Find the endpoints for the range
    if RangeType.UNIT in range_type:
        # Unit range from [0,1[
        start, end = 0., 1.0
    elif RangeType.FREQ in range_type:
        assert sw is not None, "Frequency mode requires a spectral width (sw)"
        # Frequency range from [0, sw[
        start, end = 0., sw
    elif RangeType.TIME in range_type:
        assert sw is not None, "Time mode requires a spectral width (sw)"
        # Time range from [0, sw[
        dw = sw ** -1
        group_delay = 0.0 if group_delay is None else floor(group_delay)
        tmax = dw * npts

        if RangeType.GROUP_DELAY in range_type:
            # Appli group delay, if specified
            start, end = 0. - group_delay * dw, tmax - group_delay * dw
        else:
            start, end = 0., tmax
    else:
        raise NotImplementedError

    # Center the range, if needed
    if RangeType.CENTER in range_type:
        middle = (end + start) / 2.
        start -= middle
        end -= middle

    # Invert, if needed
    if RangeType.INVERT in range_type:
        start, end = end, start
        delta = +1. * (max(start, end) - min(start, end)) / float(npts)
    else:
        delta = -1. * (max(start, end) - min(start, end)) / float(npts)

    # Generate the range
    if RangeType.ENDPOINT in range_type:
        return start, end
    else:
        return start, end + delta


def gen_range(npts: int,
              range_type: RangeType = RangeType.UNIT,
              sw: t.Optional[float] = None,
              group_delay: t.Optional[float] = None) -> torch.Tensor:
    """Generate a range of values based on spectral parameters

    Parameters
    ----------
    npts
        The number of points for the range
    range_type
        The type of range to generate
    sw
        The spectral width (in Hz) for the spectrum
    group_delay
        The optional group delay (in pts) to apply in offsetting time ranges

    Returns
    -------
    tensor_range
        A tensor with the range values
    """
    start, end = range_endpoints(npts=npts, range_type=range_type, sw=sw,
                                 group_delay=group_delay)

    # Generate the range
    return torch.linspace(start=start, end=end, steps=npts)
