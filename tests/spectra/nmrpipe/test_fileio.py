"""
Test the NMRPipe fileio functions
"""
from cmath import isclose
from pathlib import Path

import pytest
from pocketchemist_nmr.spectra.nmrpipe.fileio import (
    parse_nmrpipe_meta, load_nmrpipe_tensor, load_nmrpipe_multifile_tensor,
    save_nmrpipe_tensor)
from pocketchemist_nmr.spectra.nmrpipe.meta import load_nmrpipe_meta

from ...conftest import expected


@pytest.mark.parametrize('expected', expected().values())
def test_parse_nmrpipe_meta(expected):
    """Test the parse_nmrpipe_meta function"""
    # Load the meta dict
    if str(expected['filepath']).count('%') == 1:
        filepath = str(expected['filepath']) % 1
    else:
        filepath = str(expected['filepath'])

    with open(filepath, 'rb') as f:
        print(f"Loading '{filepath}'")
        meta = load_nmrpipe_meta(f)

    # Check the parsing of values
    result = parse_nmrpipe_meta(meta)

    # See the variable definitions from the expected function
    assert result['ndims'] == expected['header']['ndims']
    assert result['order'] == expected['header']['order']
    assert result['data_type'] == expected['header']['data_type']
    assert result['data_pts'] == expected['header']['data_pts']
    assert result['pts'] == expected['header']['pts']
    assert result['data_layout'] == expected['header']['data_layout']


@pytest.mark.parametrize('expected', expected(multifile=False).values())
def test_load_nmrpipe_tensor(expected, benchmark):
    """Test the load_nmrpipe_tensor function."""
    # Load the tensor
    print(f"Loading '{expected['filepath']}'")
    benchmark.extra_info['filepath'] = expected['filepath']
    meta, tensor = benchmark(load_nmrpipe_tensor, expected['filepath'])

    # Check the loaded tensor
    assert tensor.shape == expected['spectrum']['shape']

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['spectrum']['data_heights']:
        print(loc, data_height)
        assert isclose(tensor[loc], data_height, rel_tol=0.001)


@pytest.mark.parametrize('expected', expected(multifile=True).values())
def test_load_nmrpipe_multifile_tensor(expected, benchmark):
    """Test the load_nmrpipe_multifile_tensor function."""
    # Load the tensor
    print(f"Loading '{expected['filepath']}'")
    benchmark.extra_info['filepath'] = expected['filepath']
    meta_dict, tensor = benchmark(load_nmrpipe_multifile_tensor,
                                  expected['filepath'])

    # Show information on the tensor's memory size
    tensor_size_bytes = (tensor.storage().size() *
                         tensor.storage().element_size())
    print(f"Tensor size: {tensor_size_bytes / (1024 * 1024)} MB")

    # Check the loaded tensor
    assert tensor.shape == expected['spectrum']['shape']

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['spectrum']['data_heights']:
        print(loc, data_height)
        assert isclose(tensor[loc], data_height, rel_tol=0.001)


@pytest.mark.parametrize('expected', expected(multifile=False).values())
def test_save_nmrpipe_tensor(expected, tmpdir, benchmark):
    """Test the save_nmrpipe_tensor function."""
    # Load the tensor
    print(f"Loading '{expected['filepath']}'")
    benchmark.extra_info['filepath'] = expected['filepath']
    meta, tensor = load_nmrpipe_tensor(expected['filepath'])

    # Save the tensor to a file
    tmpfilename = Path(tmpdir) / expected['filepath'].name
    save_nmrpipe_tensor(filename=tmpfilename, meta=meta, tensor=tensor)

    # Reload the tensor and see if it's the same
    meta, tensor = benchmark(load_nmrpipe_tensor, tmpfilename)

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['spectrum']['data_heights']:
        print(loc, data_height)
        assert isclose(tensor[loc], data_height, rel_tol=0.001)

    # Delete the file
    tmpfilename.unlink()
