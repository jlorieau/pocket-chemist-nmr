"""
Utility functions for spectra
"""
import torch


def split_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Split a tensor with interleaved real/imag data in the last dimension to
    a complex tensor.

    Parameters
    ----------
    tensor
        Tensor with real interleaved data in the last dimension

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
    real, imag = torch.split(tensor,
                             int(tensor.size()[-1] / 2),
                             dim=tensor.dim() - 1)
    return torch.complex(real=real, imag=imag)


def combine_from_complex(complex_tensor: torch.Tensor) -> torch.Tensor:
    """Combine a complex tensor into a real tensor with real/imag interleaved
    dimension.

    Parameters
    ----------
    complex_tensor
        Tensor with complex data in the last dimension

    Returns
    -------
    tensor
        A real tensor with the real/imag components interleaved data in the
        last dimension.

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
