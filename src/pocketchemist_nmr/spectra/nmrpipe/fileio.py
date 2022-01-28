"""
Utilities to load NMRPipe data
"""
import typing as t
from functools import reduce
from math import isclose
from pathlib import Path

from torch import (Tensor, FloatStorage, FloatTensor, cuda, as_strided,
                   view_as_complex, complex)

from .meta import NMRPipeMetaDict, load_nmrpipe_meta
from .constants import header_size_bytes, data_size_bytes
from ..constants import DataType

__all__ = ('parse_nmrpipe_meta', 'load_nmrpipe_tensor',)


def parse_nmrpipe_meta(meta: t.Optional[NMRPipeMetaDict]) -> dict:
    """Retrieve meta data from an NMRPipe meta dict."""
    result = dict()

    # Constants used for calculations
    result['data_size_bytes'] = data_size_bytes  # bytes per data element
    result['header_size_bytes'] = header_size_bytes  # bytes for header

    # Retrieve the number of dimensions
    ndims = int(meta['FDDIMCOUNT'])
    result['ndims'] = ndims

    # Retrieve the number of points (R+I) for the last dimension
    result['fdsize'] = int(meta['FDSIZE'])

    # Data ordering
    # Retrieve the order of the dimensions (1, 2, 3, 4) vs (F1, F2, F3, F4)
    # The data will be stored as inner-outer1-outer2-...
    # inner1 - outer1
    # inner2 - outer2
    # ..
    # innerN - outer2
    result['order'] = tuple(int(meta[f'FDDIMORDER{i}'])
                            for i in range(1, result['ndims'] + 1))

    # Retrieve the data type for each dimension
    data_type = []
    for dim in result['order']:
        quadflag = meta[f'FDF{dim}QUADFLAG']
        if isclose(quadflag, 0.0):  # Complex/quadrature data
            data_type.append(DataType.COMPLEX)
        elif isclose(quadflag, 1.0):  # Real/singular data
            data_type.append(DataType.REAL)
        else:
            raise NotImplementedError
    result['data_type'] = tuple(data_type)

    # Calculate the number of complex or real or imag points in this file
    # These are ordered the same as the data
    if result['order'][0] == 1:
        result['pts'] = (result['fdsize']
                         if isclose(meta['FDSPECNUM'], 0.0) else
                         result['fdsize'], int(meta['FDSPECNUM']))
    else:
        result['pts'] = (result['fdsize']
                         if isclose(meta['FDSPECNUM'], 0.0) else
                         int(meta['FDSPECNUM']), result['fdsize'])

    # Retrieve the total number of points for each dimension (real + imag)
    # for this file
    result['data_pts'] = tuple(pts * 2
                               if data_type == DataType.COMPLEX else pts
                               for pts, data_type in zip(result['pts'],
                                                         result['data_type']))

    return result


def load_nmrpipe_tensor(filename: t.Union[str, Path],
                        meta: t.Optional[NMRPipeMetaDict] = None,
                        shared: bool = True,
                        device: t.Optional[str] = None,
                        force_gpu=False) -> Tensor:
    """Load NMRPipe data from a single spectrum file (1D or 2D).

    .. note:: The 'order' metadata attribute gives the order of dimensions
              in the dataset from inner->outer1->outer2->etc. However, the
              returned torch tensor has data ordered in reverse with
              tensor[outer2][outer1][inner]

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

    # Get the parsed values from the metadata dict
    parsed = parse_nmrpipe_meta(meta)
    ndims = parsed['ndims']  # Number of dimensions
    pts = parsed['pts']  # Number of points (Complex or Real) in data
    data_type = parsed['data_type']  # The type of data for each dimension
    data_points = parsed['data_pts']  # Number of data points in each dim
    data_size_bytes = parsed['data_size_bytes']  # size of elements in bytes
    header_size_bytes = parsed['header_size_bytes'] # size of header in bytes

    # Prepare values needed to create a tensor storage
    # We calculate the size of elements in multiples of the data size (float)
    num_elems = reduce((lambda x, y: x * y), data_points)  # number of floats
    header_elems = int(header_size_bytes
                       / data_size_bytes)  # header size in floats
    total_elems = num_elems + header_elems

    # Create the storage
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

    # Strip the header from the tensor, reshape tensor and return
    # The shape ordering has to be reverse from the number of points (pts).
    # The order of dimensions in parsed['order'] is indirect -> direct, whereas
    # the frst dimension for the tensor is the direct data
    # (contiguous fid/spectrum)
    if data_type[0] == DataType.COMPLEX:
        # Recast real/imag numbers. Data ordered as:
        # R(1) R(2) ... R(N) I(1) I(2) ... I(N)
        # Split into 2 sets, real and imag
        # split = tensor[header_elems:].reshape(int(data_points[0] / 2), 2,
        #                                       *data_points[1:])
        split = tensor[header_elems:].reshape(data_points[1], 2, int(data_points[0] / 2))
        print(pts, data_points, split.size())
        print(tensor[header_elems:])
        print(split[:,0,:])
        print(split[:,1,:])
        return complex(real=split[:, 0, :], imag=split[:, 1, :])
    else:
        return tensor[header_elems:].reshape(*pts[::-1])
