"""
Test the NMRPipe fileio functions
"""
from pathlib import Path
from collections import namedtuple

import pytest
from pocketchemist_nmr.spectra.nmrpipe.fileio import load_nmrpipe_tensor

Data = namedtuple("Data", "filename shape values")

spectra_exs = [
    Data(Path('data') / 'bruker' / 'CD20170124_av500hd_101_ubq_hsqcsi2d' /
         'hsqcetfpf3gpsi2.ft2', (1024, 368),
         (551430.50000, 493020.70000, -216286.1562)),
    Data(Path('data') / 'bruker' / 'CD20170124_av500hd_101_ubq_hsqcsi2d' /
         'test.fid', (368, 640),
         (0.0, -0.0, -761.71680)),
]


@pytest.mark.parametrize('data', spectra_exs)
def test_load_nmrpipe_tensor(data: Data):
    """Test the load_nmrpipe_tensor function."""
    # Load the tensor
    tensor = load_nmrpipe_tensor(data.filename)

    # Check the loaded tensor
    assert tensor.shape == data.shape

    if len(data.shape) == 2:
        # Values retrieved from pipe2txt.tcl
        assert tensor[0][0] == data.values[0]  # check first point
        assert tensor[1][0] == data.values[1]  # check first slice
        assert tensor[-1][-1] == data.values[2]  # check last point
