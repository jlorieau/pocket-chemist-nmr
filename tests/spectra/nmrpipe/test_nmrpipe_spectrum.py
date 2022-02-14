"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from cmath import isclose
from pathlib import Path

import pytest
import torch

from pocketchemist_nmr.spectra.nmrpipe import NMRPipeSpectrum
from pocketchemist_nmr.spectra.utils import (split_block_to_complex,
                                             split_single_to_complex,
                                             combine_single_from_complex,
                                             combine_block_from_complex)

from ...conftest import expected

#: Attributes to test
attrs = ('ndims', 'order', 'domain_type', 'data_type', 'sw', 'label',
         'sign_adjustment', 'plane2dphase')


def check_attributes(spectrum, expected):
    """Check the attributes of a spectrum"""
    for attr in attrs:
        spectrum_value = getattr(spectrum, attr)
        expected_value = expected['spectrum'][attr]

        print(f"attribute: {attr}, spectrum value: '{spectrum_value}', "
              f"expected value: '{expected_value}'")
        # Check that the values match expected
        if (hasattr(expected_value, '__iter__') and
                all(isinstance(i, float) for i in expected_value)):
            # For parameters that are lists of floats, compare each individually
            # in case there are floating point errors
            assert all([isclose(i, j) for i, j in zip(spectrum_value,
                                                      expected_value)])
        else:
            assert spectrum_value == expected_value


# Property Accessors/Mutators
@pytest.mark.parametrize("expected", expected().values())
def test_nmrpipe_spectrum_properties(expected):
    """Test the NMRPipeSpectrum accessor properties
    """
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Check the attributes
    check_attributes(spectrum, expected)


@pytest.mark.parametrize("expected", expected().values())
def test_nmrpipe_spectrum_data_layout(expected):
    """Test the NMRPipeSpectrum data_layout method
    """
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    for dim, data_type in enumerate(spectrum.data_type):
        data_layout = spectrum.data_layout(dim=dim, data_type=data_type)
        assert data_layout is expected['spectrum']['data_layout'][dim]


# # I/O methods

@pytest.mark.parametrize('expected', expected(include=(
        '1d real fid', '2d complex fid',
        '3d real spectrum')).values())
def test_nmrpipe_spectrum_load_save(expected, tmpdir):
    """Test the NMRPipeSpectrum load/save methods"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Save the spectrum
    out_filepath = Path(tmpdir) / expected['filepath'].name
    spectrum.save(out_filepath=out_filepath, overwrite=False)  # new spectrum

    # Saving without overwrite with raise an exceptions
    with pytest.raises(FileExistsError):
        spectrum.save(out_filepath=out_filepath, overwrite=False)

    # Reload the spectrum
    spectrum = NMRPipeSpectrum(out_filepath)

    # Check the attributes
    check_attributes(spectrum, expected)


# Mutators/Processing methods
@pytest.mark.parametrize('expected,expected_tp',
                         ((expected()['2d complex fid'],
                           expected()['2d complex fid (tp)']),))
def test_nmrpipe_spectrum_transpose(expected, expected_tp):
    """Test the NMRPipeSpectrum transpose method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_tp['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_tp = NMRPipeSpectrum(expected_tp['filepath'])

    # Try reversing the last 2 axes
    dims = list(range(spectrum.ndims))
    spectrum.transpose(dims[-1], dims[-2])

    # Check attributes to see if they were transposed correctly
    for attr in ('domain_type', 'data_type', 'sw', 'label'):
        print(f"spectrum1 attr: '{getattr(spectrum, attr)}', "
              f"spectrum1_tp attr: '{getattr(spectrum_tp, attr)}'")
        assert getattr(spectrum, attr) == getattr(spectrum_tp, attr)

    # Check the data shape
    assert spectrum.data.size() == spectrum_tp.data.size()

    # Check the values, row-by-row
    if spectrum.ndims > 1:
        for i, (row1, row2) in enumerate(zip(spectrum.data, spectrum_tp.data)):
            print(f"Row #{i}")
            assert tuple(row1) == tuple(row2)
    else:
        raise NotImplementedError


@pytest.mark.parametrize('expected,expected_ps',
                         ((expected()['1d real spectrum'],
                           expected()['1d real spectrum (ps)']),))
def test_nmrpipe_spectrum_phase(expected, expected_ps):
    """Test the NMRPipeSpectrum phase method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_ps['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_ps = NMRPipeSpectrum(expected_ps['filepath'])

    # Get the phase to use
    dim = spectrum_ps.order[-1]
    p0 = spectrum_ps.meta[f'FDF{dim}P0']
    p1 = spectrum_ps.meta[f'FDF{dim}P1']

    # Phase the spectrum
    spectrum.phase(p0=p0, p1=p1, discard_imaginaries=True)

    # Check the values, row-by-row
    if spectrum.ndims == 1:
        assert tuple(spectrum.data) == tuple(spectrum_ps.data)
    else:
        for i, (row1, row2) in enumerate(zip(spectrum.data, spectrum_ps.data)):
            print(f"Row #{i}")
            assert tuple(row1) == tuple(row2)


@pytest.mark.parametrize('expected', expected(include=(
        '1d real fid', '2d complex fid',
        '3d complex fid')).values())
def test_nmrpipe_spectrum_ft(expected):
    """Test the NMRPipeSpectrum ft method"""
    # Load the spectrum, if needed (cache for future tests)
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])


