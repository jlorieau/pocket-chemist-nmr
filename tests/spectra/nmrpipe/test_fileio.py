"""
Test the NMRPipe fileio functions
"""
from pathlib import Path
from cmath import isclose

import pytest
from pocketchemist_nmr.spectra.nmrpipe.fileio import (parse_nmrpipe_meta,
                                                      load_nmrpipe_tensor)
from pocketchemist_nmr.spectra.nmrpipe.meta import load_nmrpipe_meta
from pocketchemist_nmr.spectra.constants import DataType


def expected():
    """Return an iteratable of dicts with expected values"""
    return (
        # 2D spectra
        {'filepath': Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec.fid',
         'ndims': 2,  # Number of dimensions in spectrum
         'order': (2, 1),  # Data ordering of data. e.g. F1, F2
         'data_type': (DataType.COMPLEX, DataType.COMPLEX),  # Type of data
         'data_pts': (640 * 2, 184 * 2),  # Num of real + imag pts, data ordered
         'pts': (640, 184),  # Num of complex or real pts, data ordered
         'shape': (640 * 2, 368 * 1),
         'data_heights': (((0, 0), 0. + 0.j),
                          ((0, -1), 0. + 0.j),
                          ((1, 0), 0. + 0.j),
                          ((-1, -1), 0. + 0.j))},

        {'filepath': Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                     'hsqcetfpf3gpsi2.ft2',
         'ndims': 2, 'order': (1, 2),
         'data_type': (DataType.REAL, DataType.REAL),
         'data_pts': (368 * 1, 1024 * 1),
         'pts': (368, 1024),
         'shape': (1024, 368),
         'data_heights': (((0, 0), 551430.50000),
                          ((0, -1), 204368.90000),
                          ((1, 0), 493020.70000),
                          ((-1, -1), -216286.20000))},

        {'filepath': Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                     'hsqcetfpf3gpsi2_complex.ft2',
         'ndims': 2, 'order': (2, 1),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX),
         'data_pts': (1024 * 2, 368 * 2),
         'pts': (1024, 368),
         'shape': (1024 * 2, 368 * 1),
         'data_heights': (((0, 0), 551430.50000 -472929.90000j),
                          ((0, -1), 204368.80000 -486499.50000j),
                          ((1, 0), -76349.69000 +416353.10000j),
                          ((-1, -1), -216286.20000))},
    )


@pytest.mark.parametrize('expected', expected())
def test_parse_nmrpipe_tensor(expected):
    """Test the parse_nmrpipe_meta function"""
    # Load the meta dict
    meta = load_nmrpipe_meta(open(expected['filepath'], 'rb'))

    # Check the parsing of values
    result = parse_nmrpipe_meta(meta)

    # See the variable definitions from the expected function
    assert result['ndims'] == expected['ndims']
    assert result['order'] == expected['order']
    assert result['data_type'] == expected['data_type']
    assert result['data_pts'] == expected['data_pts']
    assert result['pts'] == expected['pts']


@pytest.mark.parametrize('expected', expected())
def test_load_nmrpipe_tensor(expected):
    """Test the load_nmrpipe_tensor function."""
    # Load the tensor
    tensor = load_nmrpipe_tensor(expected['filepath'])

    # Check the loaded tensor
    assert tensor.shape == expected['shape']

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['data_heights']:
        print('loc:', loc, "expected:", data_height, "actual:", tensor[loc])
        assert isclose(tensor[loc], data_height, rel_tol=0.001)
