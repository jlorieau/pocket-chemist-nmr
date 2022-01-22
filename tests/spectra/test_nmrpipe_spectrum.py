"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from pathlib import Path

import pytest
import numpy as np
import nmrglue as ng

from pocketchemist_nmr.spectra.nmr_spectrum import DomainType
from pocketchemist_nmr.spectra.nmrpipe_spectrum import (NMRPipeSpectrum,
                                                        Plane2DPhase,
                                                        SignAdjustment)

spectrum2d_exs = (Path('data') / 'bruker' /
                  'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                  'hsqcetfpf3gpsi2.ft2',)

spectrum3d_exs = (Path('data') / 'bruker' /
                  'CD20170124_av500hd_103_ubq_hncosi2d' /
                  'fid' / 'test%03d.fid',)


# Property Accessors/Mutators
@pytest.mark.parametrize("in_filepath,ndims",
                         [(spec, 2) for spec in spectrum2d_exs] +
                         [(spec, 3) for spec in spectrum3d_exs])
def test_nmrpipe_spectrum_ndims(in_filepath, ndims):
    """Test the NMRPipeSpectrum ndims property (nD)"""
    # Load the spectrum and check the number of dimensions
    spectrum = NMRPipeSpectrum(in_filepath)

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    assert spectrum.ndims == ndims

    # See if the number of dimensions is correctly set in the header
    assert int(spectrum.meta['FDDIMCOUNT']) == ndims


@pytest.mark.parametrize("in_filepath,expected_order",
                         [(spec, (1, 2)) for spec in spectrum2d_exs] +
                         [(spec, (2, 1, 3)) for spec in spectrum3d_exs])
def test_nmrpipe_spectrum_order(in_filepath, expected_order):
    """Test the NMRPipeSpectrum order property"""
    # Load the spectrum and check the number of dimensions
    spectrum = NMRPipeSpectrum(in_filepath)

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    # Check the default order
    assert spectrum.order == expected_order

    # Try changing the order
    spectrum.order = tuple(reversed(expected_order))
    assert spectrum.order == tuple(reversed(expected_order))

    # Check the header
    # ex: [2.0, 1.0, 3.0, 4.0] for a 2D
    fddimorder = ([float(i) for i in spectrum.order] +
                  [float(i) for i in range(len(expected_order) + 1, 4 + 1)])
    assert spectrum.meta['FDDIMORDER'] == fddimorder


@pytest.mark.parametrize("in_filepath", spectrum2d_exs + spectrum3d_exs)
def test_nmrpipe_spectrum_domain_type(in_filepath):
    """Test the NMRPipeSpectrum domain_type function"""
    # Load the spectrum
    spectrum = NMRPipeSpectrum(in_filepath)

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    # Check that the spectral widths are reasonable
    for dim in range(0, 5):
        if 0 < dim <= spectrum.ndims:

            if in_filepath.suffix == '.fid':
                # FIDs are in the time domain
                assert spectrum.domain_type(dim) is DomainType.TIME

                # Try changing the value
                assert (spectrum.domain_type(dim, DomainType.FREQ)
                        is DomainType.FREQ)

            else:
                # Processed spectra are in the frequency domain
                assert spectrum.domain_type(dim) is DomainType.FREQ

                # Try changing the value
                assert (spectrum.domain_type(dim, DomainType.TIME)
                        is DomainType.TIME)
        else:
            # invalid dimension number
            with pytest.raises(AssertionError):
                spectrum.domain_type(dim)

            with pytest.raises(AssertionError):
                spectrum.domain_type(dim)


@pytest.mark.parametrize("in_filepath", spectrum2d_exs + spectrum3d_exs)
def test_nmrpipe_spectrum_sw(in_filepath):
    """Test the NMRPipeSpectrum sw property"""
    # Load the spectrum and check the number of dimensions
    spectrum = NMRPipeSpectrum(in_filepath)

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    # Check that the spectral widths are reasonable
    ndims = spectrum.ndims
    sws = spectrum.sw
    assert len(sws) == ndims
    assert all(0. < sw < 100000. for sw in sws)
    for dim, sw in zip(spectrum.order, sws):
        assert spectrum.meta[f"FDF{dim}SW"] == sw

    # Try changing the spectral widths
    rev_sws = tuple(reversed(sws))
    assert rev_sws != sws

    spectrum.sw = rev_sws
    for dim, sw in zip(spectrum.order, rev_sws):
        assert spectrum.meta[f"FDF{dim}SW"] == sw


@pytest.mark.parametrize("in_filepath", spectrum2d_exs + spectrum3d_exs)
def test_nmrpipe_spectrum_sign_adjustment(in_filepath):
    """Test the NMRPipeSpectrum sign_adjustment method"""
    # Load the spectrum
    spectrum = NMRPipeSpectrum(in_filepath)

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    assert spectrum.sign_adjustment() is SignAdjustment.NONE
    assert spectrum.sign_adjustment(1) is SignAdjustment.NONE


@pytest.mark.parametrize("in_filepath", spectrum2d_exs + spectrum3d_exs)
def test_nmrpipe_spectrum_plane2dphase(in_filepath):
    """Test the NMRPipeSpectrum plane2dphase method"""
    # Load the spectrum
    spectrum = NMRPipeSpectrum(in_filepath)

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    # Check the method's value
    assert spectrum.plane2dphase is Plane2DPhase.STATES
    assert spectrum.meta['FD2DPHASE'] == 2.0

    # Try modifying the value
    spectrum.plane2dphase = Plane2DPhase.TPPI
    assert spectrum.plane2dphase is Plane2DPhase.TPPI
    assert spectrum.meta['FD2DPHASE'] == 1.0


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

    # If it's spectrum with an iterator, it has to be iterator once to
    # populate self.meta and self.dict
    if spectrum.iterator is not None:
        next(spectrum)

    # Check that the spectrum was properly setup
    assert isinstance(spectrum.meta, dict)
    assert isinstance(spectrum.data, np.ndarray)
    assert isinstance(spectrum.iterator, ng.fileio.pipe.iter3D)

    # Check the header
    assert 'FDDIMCOUNT' in spectrum.meta
    assert int(spectrum.meta['FDDIMCOUNT']) == 3

    # Try iterating
    spectrum2d = next(spectrum)
    assert type(spectrum2d) == type(spectrum)
    assert isinstance(spectrum2d.meta, dict)
    assert isinstance(spectrum2d.data, np.ndarray)

    # Check the header
    assert 'FDDIMCOUNT' in spectrum2d.meta
    assert int(spectrum2d.meta['FDDIMCOUNT']) == 3


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


# Mutators/Processing methods
