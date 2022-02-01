"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from cmath import isclose
from itertools import product

import pytest

from pocketchemist_nmr.spectra.nmrpipe import (NMRPipeSpectrum, Plane2DPhase,
                                               SignAdjustment)

from .conftest import expected


# Property Accessors/Mutators
@pytest.mark.parametrize("prop,expected",
                         product(('ndims', 'order', 'domain_type', 'sw',
                                  'label', 'sign_adjustment'),
                                 expected()))
def test_nmrpipe_spectrum_properties(prop,expected):
    """Test the NMRPipeSpectrum accessor properties"""
    # Load the spectrum, if needed (cache for future tests)
    if 'spectrum' not in expected:
        expected['spectrum'] = NMRPipeSpectrum(expected['filepath'])
    spectrum = expected['spectrum']

    # Check that the spectral widths are reasonable
    spectrum_value = getattr(spectrum, prop)
    expected_value = expected[prop]

    # Check that the values match expected
    if (hasattr(expected_value, '__iter__') and
       all(isinstance(i, float) for i in expected_value)):
        # For parameters that are lists of floats, compare each individually
        # in case there are floating point errors
        assert all(isclose(i, j) for i, j in zip(spectrum_value,
                                                 expected_value))
    else:
        assert spectrum_value == expected_value

# @pytest.mark.parametrize("in_filepath", spectrum2d_exs + spectrum3d_exs)
# def test_nmrpipe_spectrum_plane2dphase(in_filepath):
#     """Test the NMRPipeSpectrum plane2dphase method"""
#     # Load the spectrum
#     spectrum = NMRPipeSpectrum(in_filepath)
#
#     # If it's spectrum with an iterator, it has to be iterator once to
#     # populate self.meta and self.dict
#     if spectrum.iterator is not None:
#         next(spectrum)
#
#     # Check the method's value
#     assert spectrum.plane2dphase is Plane2DPhase.STATES
#     assert spectrum.meta['FD2DPHASE'] == 2.0
#
#     # Try modifying the value
#     spectrum.plane2dphase = Plane2DPhase.TPPI
#     assert spectrum.plane2dphase is Plane2DPhase.TPPI
#     assert spectrum.meta['FD2DPHASE'] == 1.0
#
#
# # I/O methods
#
# @pytest.mark.parametrize("in_filepath", spectrum2d_exs)
# def test_nmrpipe_spectrum_load_2d(in_filepath):
#     """Test the loading of a NMRPipeSpectrum (2D)"""
#     spectrum = NMRPipeSpectrum(in_filepath)
#
#     # Check that the spectrum was properly setup
#     assert isinstance(spectrum.meta, dict)
#     assert isinstance(spectrum.data, np.ndarray)
#
#     # Check the header
#     assert 'FDDIMCOUNT' in spectrum.meta
#     assert int(spectrum.meta['FDDIMCOUNT']) == 2
#
#     # Check the data
#     assert spectrum.data.shape == (1024, 368)
#
#
# @pytest.mark.parametrize("in_filepath", spectrum3d_exs)
# def test_nmrpipe_spectrum_load_3d(in_filepath):
#     """Test the loading of a NMRPipeSpectrum (3D)"""
#     spectrum = NMRPipeSpectrum(in_filepath)
#
#     # If it's spectrum with an iterator, it has to be iterator once to
#     # populate self.meta and self.dict
#     if spectrum.iterator is not None:
#         next(spectrum)
#
#     # Check that the spectrum was properly setup
#     assert isinstance(spectrum.meta, dict)
#     assert isinstance(spectrum.data, np.ndarray)
#     assert spectrum.iterator is not None
#
#     # Check the header
#     assert 'FDDIMCOUNT' in spectrum.meta
#     assert int(spectrum.meta['FDDIMCOUNT']) == 3
#
#     # Try iterating
#     spectrum2d = next(spectrum)
#     assert type(spectrum2d) == type(spectrum)
#     assert isinstance(spectrum2d.meta, dict)
#     assert isinstance(spectrum2d.data, np.ndarray)
#
#     # Check the header
#     assert 'FDDIMCOUNT' in spectrum2d.meta
#     assert int(spectrum2d.meta['FDDIMCOUNT']) == 3
#
#
# def test_nmrpipe_spectrum_load_missing_2d():
#     """Test the NMRPipeSpectrum load behavior for a missing file (2D)"""
#     # Setup a test filepath for the missing file
#     missing_filepath = Path("this_is_missing.ft2")
#     assert not missing_filepath.exists()
#
#     # Try loading this missing file
#     with pytest.raises(FileNotFoundError):
#         NMRPipeSpectrum(missing_filepath)
#
#
# @pytest.mark.parametrize("in_filepath", spectrum2d_exs)
# def test_nmrpipe_spectrum_reset_2d(in_filepath):
#     """Test the NMRPipeSpectrum reset() method (2D)"""
#     # Load the spectrum
#     spectrum = NMRPipeSpectrum(in_filepath)
#
#     # Check that stuff was loaded
#     assert str(spectrum.in_filepath) == str(in_filepath)
#     assert isinstance(spectrum.meta, dict)
#     assert len(spectrum.meta) > 0
#     assert isinstance(spectrum.data, np.ndarray)
#
#     # Reset and check that stuff is put pack in place
#     spectrum.reset()
#     assert spectrum.in_filepath is None
#     assert isinstance(spectrum.meta, dict)
#     assert len(spectrum.meta) == 0
#     assert spectrum.data is None


# Mutators/Processing methods
