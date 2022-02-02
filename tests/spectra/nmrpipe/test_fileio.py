"""
Test the NMRPipe fileio functions
"""
from cmath import isclose

import pytest
from pocketchemist_nmr.spectra.nmrpipe.fileio import (
    parse_nmrpipe_meta, load_nmrpipe_tensor, load_nmrpipe_multifile_tensor)
from pocketchemist_nmr.spectra.nmrpipe.meta import load_nmrpipe_meta

from .conftest import expected


@pytest.mark.parametrize('expected',
                         expected(exclude=('3d real spectrum (2d planes)',)))
def test_parse_nmrpipe_tensor(expected):
    """Test the parse_nmrpipe_meta function"""
    # Load the meta dict
    with open(expected['filepath'], 'rb') as f:
        print(f"Loading '{expected['filepath']}'")
        meta = load_nmrpipe_meta(f)

    # Check the parsing of values
    result = parse_nmrpipe_meta(meta)

    # See the variable definitions from the expected function
    assert result['ndims'] == expected['ndims']
    assert result['order'] == expected['order']
    assert result['data_type'] == expected['data_type']
    assert result['data_pts'] == expected['data_pts']
    assert result['pts'] == expected['pts']


@pytest.mark.parametrize('expected',
                         expected(exclude=('3d real spectrum (2d planes)',)))
def test_load_nmrpipe_tensor(expected, benchmark):
    """Test the load_nmrpipe_tensor function."""
    # Load the tensor
    print(f"Loading '{expected['filepath']}'")
    benchmark.extra_info['filepath'] = expected['filepath']
    meta, tensor = benchmark(load_nmrpipe_tensor, expected['filepath'])

    # Check the loaded tensor
    assert tensor.shape == expected['shape']

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['data_heights']:
        print(loc, data_height)
        assert isclose(tensor[loc], data_height, rel_tol=0.001)


@pytest.mark.parametrize('expected',
                         expected(include=('3d real spectrum (2d planes)',)))
def test_load_nmrpipe_multifile_tensor(expected, benchmark):
    """Test the load_nmrpipe_multifile_tensor function."""
    # Load the tensor
    benchmark.extra_info['filepath'] = expected['filepath']
    meta_dict, tensor = benchmark(load_nmrpipe_multifile_tensor,
                                  expected['filepath'])

    # Show information on the tensor's memory size
    tensor_size_bytes = (tensor.storage().size() *
                         tensor.storage().element_size())
    print(f"Tensor size: {tensor_size_bytes / (1024 * 1024)} MB")

    # Check the loaded tensor
    assert tensor.shape == expected['shape']

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['data_heights']:
        assert isclose(tensor[loc], data_height, rel_tol=0.001)
