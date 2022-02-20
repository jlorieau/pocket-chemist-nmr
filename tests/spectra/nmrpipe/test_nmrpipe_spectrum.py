"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from cmath import isclose
from pathlib import Path
import typing as t

import torch

import pytest
from pocketchemist_nmr.spectra.nmrpipe import NMRPipeSpectrum
from pocketchemist_nmr.spectra.constants import ApodizationType, RangeType

from ...conftest import expected

#: Attributes to test
attrs = ('ndims', 'order', 'domain_type', 'data_type', 'sw', 'label',
         'apodization', 'group_delay', 'correct_digital_filter',
         'sign_adjustment', 'plane2dphase')


def match_attributes(spectrum, expected):
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


def match_metas(meta1: dict, meta2: dict,
                skip: t.Optional[t.Tuple[str, ...]] = None):
    """Check that 2 meta dicts match

    Parameters
    ----------
    meta1
        The first meta dict to match
    meta2
        The second meta dict to match
    skip
        An optional list of keys to skip over when doing the comparison match
    """
    # Find keys missing from between the two meta dicts
    assert len(meta1.keys() - meta2.keys()) == 0, (
        f"The following keys are in meta1 but not meta2: "
        f"{meta1.keys() - meta2.keys()}")
    assert len(meta2.keys() - meta1.keys()) == 0, (
        f"The following keys are in meta2 but not meta1: "
        f"{meta2.keys() - meta1.keys()}")

    # Check the values
    unmatched_values = dict()
    for k in meta1.keys():
        # Skip entry, if specified
        if skip is not None and k in skip:
            continue

        value1, value2 = meta1[k], meta2[k]

        # Check that the types match
        if type(value1) != type(value2):
            unmatched_values[k] = ('mismatched types', value1, value2)

        if 'MIN' in k or 'MAX' in k:
            if not isclose(value1, value2, rel_tol=0.0001):
                # For FDMAX/FDMIN/FDDISPMAX/FDDISPMIN, these should be within
                # error (or within 0.01%)
                unmatched_values[k] = ('mismatched values', value1, value2)
        elif (isinstance(value1, float) and
              round(value1, 1) != round(value2, 1)):
            # For other floats, round them to make sure they match within the
            # first decimal
            unmatched_values[k] = ('mismatched values', value1, value2)
        elif value1 != value2:
            # For other values, just check them directly
            unmatched_values[k] = ('mismatched values', value1, value2)

    # Assert that there are no unmatched_values
    if len(unmatched_values) > 0:
        msg = "The following values do not match:\n"
        msg += '\n'.join(f'  {k}: {value1} != {value2}  ({reason})'
                         for k, (reason, value1, value2)
                         in unmatched_values.items())
        raise AssertionError(msg)


# Property Accessors/Mutators
@pytest.mark.parametrize("expected", expected().values())
def test_nmrpipe_spectrum_properties(expected):
    """Test the NMRPipeSpectrum accessor properties"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Check the attributes
    match_attributes(spectrum, expected)


@pytest.mark.parametrize("expected", expected().values())
def test_nmrpipe_spectrum_data_layout(expected):
    """Test the NMRPipeSpectrum data_layout method"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    for dim, data_type in enumerate(spectrum.data_type):
        data_layout = spectrum.data_layout(dim=dim, data_type=data_type)
        assert data_layout is expected['spectrum']['data_layout'][dim]


# I/O methods

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
    match_attributes(spectrum, expected)


# Mutators/Processing methods
@pytest.mark.parametrize('expected,expected_em',
                         ((expected()['1d complex fid'],
                           expected()['1d complex fid (em)']),))
def test_nmrpipe_spectrum_apodization_exp(expected, expected_em):
    """Test the NMRPipeSpectrum apod_exponential method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_em['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_em = NMRPipeSpectrum(expected_em['filepath'])

    # Get the apodization parameters
    dim = spectrum_em.order[0]
    code = spectrum_em.meta[f'FDF{dim}APODCODE']
    lb = spectrum_em.meta[f'FDF{dim}APODQ1']

    assert code == 2.0  # apodization code for EM

    # Check that an apodization hasn't been applied yet
    assert all(apod is ApodizationType.NONE for apod in spectrum.apodization)

    # Apodization the original dataset
    spectrum.apodization_exp(lb=lb, range_type=RangeType.TIME)

    # Check the header
    match_metas(spectrum.meta, spectrum_em.meta)

    # Check the header values
    assert spectrum.apodization[0] == ApodizationType.EXPONENTIAL
    assert spectrum.meta[f'FDF{dim}APODCODE'] == 2.0
    assert isclose(spectrum.meta[f'FDF{dim}APODQ1'], lb)

    # Find the tolerance for float errors
    tol = spectrum.data.real.max() * 0.0001

    # Check the values
    if spectrum.ndims == 1:
        for row, (i, j) in enumerate(zip(spectrum.data, spectrum_em.data)):
            print(f"Row #{row}: {i}, {j}")
            assert isclose(i, j, abs_tol=tol)
    else:
        raise NotImplementedError


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

    # Check the header
    match_metas(spectrum.meta, spectrum_tp.meta)

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
    spectrum.phase(p0=p0, p1=p1, range_type=RangeType.UNIT,
                   discard_imaginaries=True)

    # Check the header
    match_metas(spectrum.meta, spectrum_ps.meta)

    # Find a tolerance for matching numbers. The numbers do not exactly match
    # the reference dataset due to rounding errors (presumably)
    tol = max(abs(spectrum.data.max()), abs(spectrum.data.min())) * 0.0001

    # Check the values, row-by-row
    if spectrum.ndims == 1:
        for count, (i, j) in enumerate(zip(spectrum.data, spectrum_ps.data)):
            assert isclose(i, j, abs_tol=tol)
    else:
        for i, (row1, row2) in enumerate(zip(spectrum.data, spectrum_ps.data)):
            print(f"Row #{i}")
            assert all(isclose(i, j) for i, j in zip(row1, row2))


@pytest.mark.parametrize('expected,expected_ft',
                         ((expected()['1d complex fid'],
                           expected()['1d complex fid (ft)']),))
def test_nmrpipe_spectrum_ft(expected, expected_ft):
    """Test the NMRPipeSpectrum ft method"""
    # Load the spectrum, if needed (cache for future tests)
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_ft['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_ft = NMRPipeSpectrum(expected_ft['filepath'])

    # Conduct the Fourier transform
    spectrum.ft()

    # Check the header. The FT spectrum does not exactly match the reference
    # spectrum (spectrum_ft), so the following values are skipped. The
    # intensities without phasing differences are compared by the power spectrum
    # below
    match_metas(spectrum.meta, spectrum_ft.meta,
                skip=('FDMAX', 'FDMIN', 'FDDISPMAX', 'FDDISPMIN'))

    # Compare the power spectra, which are phase insensitive
    pow_spectrum = torch.sqrt(spectrum.data.real ** 2 +
                              spectrum.data.imag ** 2)
    pow_spectrum_ft = torch.sqrt(spectrum_ft.data.real ** 2 +
                                 spectrum_ft.data.imag ** 2)

    # Find a tolerance for matching numbers. The numbers do not exactly match
    # the reference dataset due to rounding errors (presumably)
    tol = max(abs(pow_spectrum_ft.max()), abs(pow_spectrum_ft.min())) * 0.00001

    if spectrum.ndims == 1:
        # Check the normalized values, row-by-row
        for count, (i, j) in enumerate(zip(pow_spectrum, pow_spectrum_ft)):
            print(f"Row#{count}: "
                  f"{i}, {j}")
            assert isclose(i, j, abs_tol=tol)
    else:
        raise NotImplementedError



