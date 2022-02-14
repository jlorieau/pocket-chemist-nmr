"""
Utilities to load NMRPipe data
"""
import typing as t
from functools import reduce
from itertools import zip_longest
from math import isclose
from pathlib import Path

import torch

from .meta import NMRPipeMetaDict, load_nmrpipe_meta, save_nmrpipe_meta
from .constants import header_size_bytes, data_size_bytes
from ..constants import DataType
from ..utils import split_block_to_complex, combine_block_from_complex

__all__ = ('parse_nmrpipe_meta', 'load_nmrpipe_tensor',
           'load_nmrpipe_multifile_tensor')


# Single file I/O

def parse_nmrpipe_meta(meta: t.Optional[NMRPipeMetaDict]) -> dict:
    """Retrieve meta data from an NMRPipe meta dict.

    Data is ordered according to NMRPipe ordering, which has the
    inner-outer1-outer2 order.
    """
    result = dict()

    # Constants used for calculations
    result['data_size_bytes'] = data_size_bytes  # bytes per data element
    result['header_size_bytes'] = header_size_bytes  # bytes for header

    # Retrieve the number of dimensions
    ndims = int(meta['FDDIMCOUNT'])
    result['ndims'] = ndims

    # Retrieve the number of points (complex or real) for the last dimension
    # (i.e. the current dimension)
    result['fdsize'] = int(meta['FDSIZE'])

    # Retrieve the number of 1Ds in the file (real + imag)
    result['fdspecnum'] = int(meta['FDSPECNUM'])

    # Retrieve the interleave/block of the last dimension
    result['fdslicecount0'] = int(meta['FDSLICECOUNT0'])
    result['fdslicecount1'] = int(meta['FDSLICECOUNT1'])

    # Retrieve the number of planes (3D, 4D) in the file (real + imag)
    result['fdf3size'] = int(meta['FDF3SIZE'])
    result['fdf4size'] = int(meta['FDF4SIZE'])
    result['fdfilecount'] = int(meta['FDFILECOUNT'])

    # Data ordering
    # Retrieve the order of the dimensions (1, 2, 3, 4) vs (F1, F2, F3, F4)
    # The data will be stored as inner-outer1-outer2-...
    # inner1 - outer1
    # inner2 - outer2
    # ..
    # innerN - outer2
    result['order'] = tuple(int(meta[f'FDDIMORDER{i}'])
                            for i in range(1, ndims + 1))

    # Retrieve the data type for each dimension. e.g. real, complex
    # These are ordered the same as the data (result['order'])
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

    # Data ordered points
    # Calculate the number of complex or real or imag points in this file, as
    # ordered in the data -- i.e. same order as result['order']
    pts = []  # Complex OR Real points
    data_pts = []  # Real + Imag points

    for dim, data_type, label in zip_longest(
            result['order'], result['data_type'],
            ('fdsize', 'fdspecnum', 'fdf3size', 'fdf4size'), fillvalue=None):
        # If the dimension hasn't been assigned, quit processing
        if dim is None:
            break
        if label in ('fdf1size', 'fdf2size'):
            continue

        # Get the value for the variable
        value = result[label]

        # Assign the number of points (pts) and number of data points (data_pts)
        if label == 'fdsize':
            # FDDIZE contains the number of points (Complex or Real)
            pts.append(value)
            data_pts.append(value * 2 if data_type == DataType.COMPLEX else
                            value)
        elif label in ('fdspecnum', 'fdf3size', 'fdf4size'):
            # FDSPECNUM contains the number of data points in dimension 2
            # (Real + Imag)
            # FDF3SIZE/FDF4SIZE contains the number of data points in
            # dimensions 3 and 4 (Real + Imag)
            pts.append(int(value / 2) if data_type == DataType.COMPLEX else
                       value)
            data_pts.append(value)
        else:
            raise NotImplementedError

    # Add the point entries to the result dict
    result['pts'] = tuple(pts)
    result['data_pts'] = tuple(data_pts)

    return result


def load_nmrpipe_tensor(filename: t.Union[str, Path],
                        meta: t.Optional[NMRPipeMetaDict] = None,
                        shared: bool = True,
                        device: t.Optional[str] = None,
                        force_gpu=False) -> (dict, torch.Tensor):
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
    meta, tensor
        The metadata dict and tensor for the spectrum's data
    """
    # Check that the file exists
    if not Path(filename).exists():
        raise FileNotFoundError(
            f"Could not find file for file path '{filename}'")

    # Load the meta dict, if needed
    if meta is None:
        with open(filename, 'rb') as f:
            meta = load_nmrpipe_meta(f)

    # Get the parsed values from the metadata dict
    parsed = parse_nmrpipe_meta(meta)
    ndims = parsed['ndims']  # Number of dimensions
    points = parsed['pts']  # Number of points (Complex or Real) in data
    data_type = parsed['data_type']  # The type of data for each dimension
    data_points = parsed['data_pts']  # Number of data points in each dim
    data_size_bytes = parsed['data_size_bytes']  # size of elements in bytes
    header_size_bytes = parsed['header_size_bytes'] # size of header in bytes

    # Get data relevant for 3Ds and 4Ds
    file_count = parsed['fdfilecount']  # Num of files spectrum is split over
    f3size = parsed['fdf3size']  # Num of points in F3
    f4size = parsed['fdf4size']  # Num of points in F3

    # For spectra split over multiple files, reduce the dimensionality of
    # points/data_points for over multiple files
    if file_count > 1:
        # Reduce the number of points if the last dimension corresponds to
        # the number of files in which the spectrum is split
        if data_points[-1] == file_count or data_points[-1] == file_count:
            points = points[:-1]  # Remove last point
            data_points = data_points[:-1]  # Remove last point

        # For spectra in which 2 dimensions (i.e. 4Ds) are split over multiple
        # files, then a product of dimensions is needed
        elif (len(data_points) > 2 and
              data_points[-1] * data_points[-2] == file_count):
            points = points[:-2]  # Remove last 2 points
            data_points = data_points[:-2]  # Remove last 2 points

    # Prepare values needed to create a tensor storage
    # We calculate the size of elements in multiples of the data size (float)
    num_elems = reduce((lambda x, y: x * y), data_points)  # number of floats
    header_elems = int(header_size_bytes
                       / data_size_bytes)  # header size in floats
    total_elems = num_elems + header_elems

    # Create the storage
    if torch.cuda.is_available() or force_gpu:
        # Allocate on the GPU
        storage = torch.FloatStorage.from_file(str(filename), shared=shared,
                                               size=total_elems)
        storage = storage.cuda(device=device)

    else:
        # Allocate on CPU
        storage = torch.FloatStorage.from_file(str(filename), shared=shared,
                                               size=total_elems)

    # Create the tensor
    tensor = (torch.FloatTensor(storage) if device is None else
              torch.FloatStorage(storage, device=device))

    # Strip the header
    tensor = tensor[header_elems:]

    # Strip the header from the tensor, reshape tensor and return tensor
    # The shape ordering has to be reversed from the number of points (pts).
    # NMRPipe data: inner->outer1->outer2
    # Tensor data: outer2->outer1->inner
    if data_type[0] == DataType.COMPLEX:
        # Recast real/imag numbers. Data ordered as:
        # R(1) R(2) ... R(N) I(1) I(2) ... I(N)
        return meta, split_block_to_complex(tensor.reshape(data_points[::-1]))
    else:
        return meta, tensor.reshape(*data_points[::-1])


def load_nmrpipe_multifile_tensor(filemask: str,
                                  meta: t.Optional[dict] = None,
                                  shared: bool = True,
                                  device: t.Optional[str] = None,
                                  force_gpu: bool = False) \
        -> (t.List[dict], torch.Tensor):
    """Load NMRPipe data from a spectrum over multiple files.

    .. note:: The 'order' metadata attribute gives the order of dimensions
              in the dataset from inner->outer1->outer2->etc. However, the
              returned torch tensor has data ordered in reverse with
              tensor[outer2][outer1][inner]

    Parameters
    ----------
    filemask
        The filemask for the NMRPipe spectrum. e.g. fid/test%03d.fid for
        fid/test001.fid, fid/test002.fid, etc.
    order
        The order to load the data. If 'default', the data is loaded in the
        same order as the data saved on disk
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
    metas, tensor
        The metadata dicts and tensor for the spectrum's data
    """
    # Convert filemasks into a listing of filenames for existing files
    filepaths = []
    if str(filemask).count("%") == 1:  # ex: test001.fid
        for i in range(1, 10000):
            filepath = Path(str(filemask) % i)
            if not filepath.exists():
                break
            filepaths.append(filepath)
    elif str(filemask).count("%") == 2:   # ex: test001_001.fid
        for i in range(1, 10000):
            missing_j = True
            for j in range(1, 10000):
                filepath = Path(str(filemask) % i, j)
                if not filepath.exists():
                    break
                else:
                    missing_j = False
                filepaths.append(filepath)
            if missing_j:
                break
    else:
        raise NotImplementedError

    if len(filepaths) == 0:
        raise FileNotFoundError(
            f"Could not find files that matched the file mask '{filemask}'")

    # Concatenate tensors
    datasets = tuple(load_nmrpipe_tensor(filepath, meta=meta, shared=shared,
                                         device=device, force_gpu=force_gpu)
                     for filepath in filepaths)
    meta_dicts = [meta for meta, _ in datasets]
    tensor = torch.stack(tuple(data for _, data in datasets))
    return meta_dicts, tensor


def save_nmrpipe_tensor(filename: t.Union[str, Path],
                        meta: NMRPipeMetaDict,
                        tensor: torch.Tensor,
                        overwrite=True):
    """Save a tensor in a single file in NMRPipe format."""
    if Path(filename).exists() and not overwrite:
        raise FileExistsError

    # Unpack the real/imag components
    if tensor.is_complex():
        tensor = combine_block_from_complex(tensor)

    # Update meta dict entries, as needed
    meta['FDMAX'] = torch.max(tensor)
    meta['FDDISPMAX'] = meta['FDMAX']
    meta['FDMIN'] = torch.min(tensor)
    meta['FDDISPMIN'] = meta['FDMIN']

    with open(filename, 'wb') as f:
        # Save the header
        data = save_nmrpipe_meta(meta=meta)
        f.write(data)

        # Create a flattend view of the tensor
        flatten = tensor.flatten()

        # Save the data in inner-outer1-outer2 order
        flatten.numpy().tofile(f)
