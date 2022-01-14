"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from pathlib import Path

import pytest
import numpy as np
import nmrglue as ng

from pocketchemist_nmr.spectra import NMRPipeSpectrum

spectrum2d_exs = (Path('data') / 'bruker' /
                  'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                  'hsqcetfpf3gpsi2.ft2',)

spectrum3d_exs = (Path('data') / 'bruker' /
                  'CD20170124_av500hd_103_ubq_hncosi2d' /
                  'fid' / 'test%03d.fid',)

# I/O methods

@pytest.mark.parametrize("in_filepath", spectrum2d_exs)
def test_nmrpipe_spectrum_load_2d(in_filepath):
    """Test the loading of a NMRPipeSpectrum (2D)"""
    spectrum = NMRPipeSpectrum(in_filepath)

    # Check that the spectrum was properly setup
    assert isinstance(spectrum.meta, dict)
    assert isinstance(spectrum.data, np.ndarray)

    # Check the header
    assert 'FDDIMCOUNT' in spectrum.meta
    assert int(spectrum.meta['FDDIMCOUNT']) == 2

    # Check the data
    assert spectrum.data.shape == (1024, 368)


@pytest.mark.parametrize("in_filepath", spectrum3d_exs)
def test_nmrpipe_spectrum_load_3d(in_filepath):
    """Test the loading of a NMRPipeSpectrum (3D)"""
    spectrum = NMRPipeSpectrum(in_filepath)

    # Check that the spectrum was properly setup
    assert isinstance(spectrum.meta, dict)
    assert isinstance(spectrum.data, ng.fileio.pipe.iter3D)

    # Check the header
    assert 'FDDIMCOUNT' in spectrum.meta
    assert int(spectrum.meta['FDDIMCOUNT']) == 3


def test_nmrpipe_spectrum_load_missing_2d():
    """Test the NMRPipeSpectrum load behavior for a missing file (2D)"""
    # Setup a test filepath for the missing file
    missing_filepath = Path("this_is_missing.ft2")
    assert not missing_filepath.exists()

    # Try loading this missing file
    with pytest.raises(FileNotFoundError):
        NMRPipeSpectrum(missing_filepath)


@pytest.mark.parametrize("in_filepath", spectrum2d_exs)
def test_nmrpipe_spectrum_reset_2d(in_filepath):
    """Test the NMRPipeSpectrum reset() method (2D)"""
    # Load the spectrum
    spectrum = NMRPipeSpectrum(in_filepath)

    # Check that stuff was loaded
    assert str(spectrum.in_filepath) == str(in_filepath)
    assert isinstance(spectrum.meta, dict)
    assert len(spectrum.meta) > 0
    assert isinstance(spectrum.data, np.ndarray)

    # Reset and check that stuff is put pack in place
    spectrum.reset()
    assert spectrum.in_filepath is None
    assert isinstance(spectrum.meta, dict)
    assert len(spectrum.meta) == 0
    assert spectrum.data is None


# Accessors/Mutators

@pytest.mark.parametrize("in_filepath", spectrum2d_exs)
def test_nmrpipe_spectrum_ndims_2d(in_filepath):
    """Test the NMRPipeSpectrum ndims property (2D)"""
    # Load the spectrum and check the number of dimensions
    spectrum = NMRPipeSpectrum(in_filepath)
    assert spectrum.ndims == 2

    # Try removing the header entry and see if the number of dimensions is
    # correctly read out
    del spectrum.meta['FDDIMCOUNT']
    assert spectrum.ndims == 2


@pytest.mark.parametrize("in_filepath", spectrum2d_exs)
def test_nmrpipe_spectrum_ndims_2d(in_filepath):
    """Test the NMRPipeSpectrum order property (2D)"""
    # Load the spectrum and check the number of dimensions
    spectrum = NMRPipeSpectrum(in_filepath)

    # Check the default order
    assert spectrum.order == (1, 2)

    # Try changing the order
    spectrum.order = (2, 1)
    assert spectrum.order == (2, 1)
    assert spectrum.meta['FDDIMORDER'] == [2.0, 1.0, 3.0, 4.0]
