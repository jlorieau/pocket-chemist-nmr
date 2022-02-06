"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from cmath import isclose
from pathlib import Path

import pytest

from pocketchemist_nmr.spectra.nmrpipe import NMRPipeSpectrum
from pocketchemist_nmr.spectra.constants import DataType

from .conftest import expected

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

@pytest.mark.parametrize('expected', expected(include=(
        '1d real fid', '2d complex fid',
        '3d real/real/complex spectrum (2d planes)')).values())
def test_nmrpipe_spectrum_permute(expected):
    """Test the NMRPipeSpectrum permute method"""
    # Load the spectrum, if needed (cache for future tests)
    print(f"Loading spectrum '{expected['filepath']}")
    if 'loaded_spectrum' not in expected:
        expected['loaded_spectrum'] = NMRPipeSpectrum(expected['filepath'])
    spectrum = expected['loaded_spectrum']

    # 1d spectra cannot be permuted
    if spectrum.ndims == 1:
        with pytest.raises(AssertionError):
            spectrum.permute((1,))
        return None

    # Copy old_values to compare to
    old = {attr: getattr(spectrum, attr) for attr in ('domain_type',
                                                      'data_type', 'sw',
                                                      'label',
                                                      'sign_adjustment')}
    old_size = list(spectrum.data.size())

    # Try reversing the axes. e.g. (0, 1, 2) -> (2, 1, 0)
    spectrum.permute(tuple(range(spectrum.ndims)[::-1]))

    # Check that the data was correctly updated
    for attr, old_value in old.items():
        new_value = getattr(spectrum, attr)
        assert old_value == new_value[::-1]  # reverse order

    # The tensor size is reversed and may have increased or decreased depending
    # on whether a complex dimension was split or stacked
    if old['data_type'][-1] == DataType.COMPLEX:
        old_size[-1] = int(old_size[-1] * 2)
    if spectrum.data_type[-1] == DataType.COMPLEX:
        old_size[0] = int(old_size[0] / 2)

    assert spectrum.data.size() == tuple(old_size)[::-1]
