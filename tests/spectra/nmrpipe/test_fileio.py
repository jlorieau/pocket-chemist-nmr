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
        # 1D spectra
        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_100_ubq_oneone1d' / 'spec.fid',
         'ndims': 1,  # Number of dimensions in spectrum
         'order': (2,),  # Order of data
         'data_type': (DataType.COMPLEX,),  # Type of data
         'data_pts': (799 * 2,),
         'pts': (799,),
         'shape': (799 * 1,),
         'data_heights': (((0,), 0. + 0.j),
                          ((-1,), -359985.70000 - 16418.97000j))},

        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_100_ubq_oneone1d' / 'oneone-echo_N-dcpl.jll.ft',
         'ndims': 1,
         'order': (2,),
         'data_type': (DataType.REAL,),
         'data_pts': (8192 * 1,),
         'pts': (8192,),
         'shape': (8192 * 1,),
         'data_heights': (((0,), 491585.80000),
                          ((-1,), 594718.70000))},

        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_100_ubq_oneone1d' /
         'oneone-echo_N-dcpl.jll_complex.ft',
         'ndims': 1,
         'order': (2,),
         'data_type': (DataType.COMPLEX,),
         'data_pts': (8192 * 2,),
         'pts': (8192,),
         'shape': (8192 * 1,),
         'data_heights': (((0,), 491585.80000 - 1010224.00000j),
                          ((-1,), 594718.70000 - 968423.10000j)),},

        # 2D spectra
        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec.fid',
         'ndims': 2,  # Number of dimensions in spectrum
         # Data ordering of data. (direct, indirect) e.g. F1, F2
         'order': (2, 1),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX),  # Type of data
         'data_pts': (640 * 2, 184 * 2),  # Num of real + imag pts, data ordered
         'pts': (640, 184),  # Num of complex or real pts, data ordered
         # Shape of returned tensor (indirect, direct), reverse of pts
         'shape': (184 * 2, 640 * 1),
         'data_heights': (((0, 0), 0. + 0.j),
                          ((0, -1), 5619.12600 - 4132.80200j),
                          ((1, 0), 0. + 0.j),
                          ((-1, -1), -761.71680 - 996.09120j))},

        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'hsqcetfpf3gpsi2.ft2',
         'ndims': 2, 'order': (1, 2),
         'data_type': (DataType.REAL, DataType.REAL),
         'data_pts': (368 * 1, 1024 * 1),
         'pts': (368, 1024),
         'shape': (1024 * 1, 368 * 1),
         'data_heights': (((0, 0), 551430.50000),
                          ((0, -1), 204368.90000),
                          ((1, 0), 493020.70000),
                          ((-1, -1), -216286.20000))},

        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'hsqcetfpf3gpsi2_complex.ft2',
         'ndims': 2, 'order': (2, 1),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX),
         'data_pts': (1024 * 2, 368 * 2),
         'pts': (1024, 368),
         'shape': (368 * 2, 1024 * 1),
         'data_heights': (((0, 0), 551430.50000 - 76349.69000j),
                          ((0, -1), 73015.44000 + 445419.70000j),
                          ((1, 0), -472929.90000 + 416353.10000j),
                          ((-1, -1), -136615.50000 - 991015.50000j))},

        # 3D Spectra
        {'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_103_ubq_hncosi2d' / 'fid' / 'spec001.fid',
         'ndims': 3, 'order': (2, 1, 3),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX, DataType.COMPLEX,),
         'data_pts': (559 * 2, 39 * 2, 51 * 2),
         'pts': (559, 39, 51),
         'shape': (39 * 2, 559),
         'data_heights': (((0, 0), 0. + 0.j),
                          ((0, -1), -6837.88700 + 5389.64600j),
                          ((1, 0), 0. + 0.j),
                          ((-1, -1), -676.75750 + 13228.51000j,))},
    )


@pytest.mark.parametrize('expected', expected())
def test_parse_nmrpipe_tensor(expected):
    """Test the parse_nmrpipe_meta function"""
    # Load the meta dict
    with open(expected['filepath'], 'rb') as f:
        meta = load_nmrpipe_meta(f)

    # Check the parsing of values
    result = parse_nmrpipe_meta(meta)

    # See the variable definitions from the expected function
    assert result['ndims'] == expected['ndims']
    assert result['order'] == expected['order']
    assert result['data_type'] == expected['data_type']
    assert result['data_pts'] == expected['data_pts']
    assert result['pts'] == expected['pts']


@pytest.mark.parametrize('expected', expected())
def test_load_nmrpipe_tensor(expected, benchmark):
    """Test the load_nmrpipe_tensor function."""
    # Load the tensor
    benchmark.extra_info['filepath'] = expected['filepath']
    tensor = benchmark(load_nmrpipe_tensor, expected['filepath'])

    # Check the loaded tensor
    assert tensor.shape == expected['shape']

    # Check the data values for some key points (locations) in the data
    for loc, data_height in expected['data_heights']:
        print('loc:', loc, "expected:", data_height, "actual:", tensor[loc])
        assert isclose(tensor[loc], data_height, rel_tol=0.001)
