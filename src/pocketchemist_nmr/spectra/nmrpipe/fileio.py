"""
Utilities to load NMRPipe data
"""
import typing as t
from functools import reduce
from pathlib import Path

from torch import FloatStorage, FloatTensor, cuda

from .meta import NMRPipeMetaDict, load_nmrpipe_meta, header_size
from ..nmr_spectrum import DomainType

__all__ = ('load_nmrpipe_tensor',)

#: The number of bytes per float
float_bytes = 4


def load_nmrpipe_tensor(filename: t.Union[str, Path],
                        meta: t.Optional[NMRPipeMetaDict] = None,
                        shared: bool = True,
                        device: t.Optional[str] = None,
                        force_gpu=False) -> FloatTensor:
    """Load NMRPipe data from a single spectrum file (1D or 2D).

    Parameters
    ----------
    filename
        The filename for the NMRPipe 1D or 2D spectrum
    meta
        The NMRPipe metadata dict
    shared
        Create the tensor storage to be shared between threads/processing
    device
        The name of the device to allocate the memory on.
    force_gpu
        Force allocating the tensor on the GPU

    Returns
    -------
    tensor
        The tensor for the spectrum's data
    """
    # Load the meta dict, if needed
    if meta is None:
        with open(filename, 'rb') as f:
            meta = load_nmrpipe_meta(f)

    # Get the spectrum's size
    ndims = int(meta['FDDIMCOUNT'])  # Number of dimensions
    fdsize = int(meta['FDSIZE'])  # Number of points in the last dimension

    # Geta meta data on the spectrum to properly load the file
    if ndims == 2:
        fdspecnum = int(meta['FDSPECNUM'])  # Number of points in the other dim

        # Get data ordering for the F1/F2 dimensions
        order = int(meta['FDDIMORDER1']), int(meta['FDDIMORDER2'])

        # Get the type of data (real, complex, imag)
        domain_types = tuple(meta[f'FDF{dim}QUADFLAG'] for dim in order)

        shape = (fdspecnum, fdsize)
    elif ndims == 1:
        # Get data ordering for the F1 dimension
        order = (int(meta['FDIMORDER1']),)

        # Get the type of data (real, complex, imag)
        domain_types = (meta[f'FDF{order}QUADFLAG'],)

        shape = (fdsize,)
    else:
        raise NotImplementedError

    # Create the storage
    num_elems = reduce((lambda x, y: x * y), shape)  # data size (in float size)
    hdr_elems = int(header_size / float_bytes)  # header size (in float size)
    total_elems = num_elems + hdr_elems

    if cuda.is_available() or force_gpu:
        # Allocate on the GPU
        storage = FloatStorage.from_file(str(filename), shared=shared,
                                         size=total_elems).cuda(device=device)

    else:
        # Allocate on CPU
        storage = FloatStorage.from_file(str(filename), shared=shared,
                                         size=total_elems)

    # Create the tensor
    tensor = FloatTensor(storage, device=device)

    # Return the tensor offset by the size of the header and reshaped to match
    # the dimensionality of the data
    return tensor[hdr_elems:].reshape(*shape)
