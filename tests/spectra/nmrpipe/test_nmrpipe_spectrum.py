"""
Test the spectra/nmrpipe_spectrum.py submodule
"""
from cmath import isclose
from math import floor
from pathlib import Path
from itertools import product, chain
import typing as t

import torch
import pytest
from pytest_cases import parametrize_with_cases, get_all_cases
from pocketchemist_nmr.spectra.nmrpipe import NMRPipeSpectrum
from pocketchemist_nmr.spectra.constants import (ApodizationType, RangeType,
                                                 UnitType)

#: Attributes to test
attrs = ('ndims', 'order', 'domain_type', 'data_type', 'sw_hz', 'sw_ppm',
         'car_hz', 'car_ppm', 'range_hz', 'range_ppm', 'range_s', 'obs_mhz',
         'label', 'apodization', 'group_delay', 'correct_digital_filter',
         'sign_adjustment', 'plane2dphase')


def parametrize_casesets(*globs, cases=None, prefix='data_') -> tuple:
    """Convert a serirs of case globs into a set of cases for parametrization.
    """
    # Convert globs to functions
    funcs = []
    dummy = lambda: None
    for glob in globs:
        glob_funcs = map(lambda glob: get_all_cases(dummy, cases=cases,
                                                    prefix=prefix, glob=glob),
                         glob if not isinstance(glob, str) else (glob,))
        glob_funcs = chain.from_iterable(glob_funcs)
        funcs.append(glob_funcs)

    # Create a generator for the product of these
    return tuple(tuple(f() for f in prod) if len(prod) > 1 else prod[0]()
                 for prod in product(*funcs))


def match_attributes(spectrum, expected):
    """Check the attributes of a spectrum"""
    unmatched_values = dict()

    for attr in attrs:
        spectrum_value = getattr(spectrum, attr)
        expected_value = expected['spectrum'][attr]

        # Check that the values match expected
        if (hasattr(expected_value, '__iter__') and
                all(isinstance(i, float) for i in expected_value)):
            # Float parameters
            if not all(isclose(i, j)
                       for i, j in zip(spectrum_value, expected_value)):
                unmatched_values[attr] = ('mismatched float values',
                                          spectrum_value, expected_value)
        elif (hasattr(expected_value, '__iter__') and
              all(isinstance(i, tuple) for i in expected_value) and
              all(isinstance(j, float) for i in expected_value for j in i )):
            # Tuple of tuples of floats
            if not all(isclose(i, j)
                       for tpl1, tpl2 in zip(spectrum_value, expected_value)
                       for i, j in zip(tpl1, tpl2)):
                unmatched_values[attr] = ('mistmatched tuple of tuple floats',
                                          spectrum_value, expected_value)

        else:
            # All other values are compared directly
            if not spectrum_value == expected_value:
                unmatched_values[attr] = ('mistmatched values',
                                          spectrum_value, expected_value)

    # Assert that there are no unmatched_values
    if len(unmatched_values) > 0:
        msg = "The following values (spectrum vs expected) do not match:\n"
        msg += '\n'.join(f'  {k}: {value1} != {value2}  ({reason})'
                         for k, (reason, value1, value2)
                         in unmatched_values.items())
        raise AssertionError(msg)


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
                unmatched_values[k] = ('mismatched min/max values', value1,
                                       value2)
        elif isinstance(value1, float):
            if round(value1, 2) != round(value2, 2):
                # For other floats, round them to make sure they match within
                # the first decimal
                unmatched_values[k] = ('mismatched float values',
                                       value1, value2)
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


def match_tuple_floats(t1, t2, **kwargs):
    """Match the values of two tuples with floats"""
    mismatched = dict()
    for item, (value1, value2) in enumerate(zip(t1, t2)):
        if not isclose(value1, value2, **kwargs):
            mismatched[item] = value1, value2

    if len(mismatched) > 0:
        msg = f"Mismatch between tuple of floats {t1} and {t2}:\n"
        msg += '\n'.join(f'  item {item}: {value1} != {value2}'
                         for item, (value1, value2) in
                         sorted(mismatched.items()))
        raise AssertionError(msg)


# Property Accessors/Mutators
@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='...cases.nmrpipe')
def test_nmrpipe_spectrum_properties(expected):
    """Test the NMRPipeSpectrum accessor properties"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Configure the range types to match NMRPipe's processing
    spectrum.freq_range_type = RangeType.FREQ
    spectrum.time_range_type = RangeType.TIME
    spectrum.unit_range_type = RangeType.UNIT

    # Check the attributes
    match_attributes(spectrum, expected)


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='...cases.nmrpipe')
def test_nmrpipe_spectrum_data_layout(expected):
    """Test the NMRPipeSpectrum data_layout method"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    for dim, data_type in enumerate(spectrum.data_type):
        data_layout = spectrum.data_layout(dim=dim, data_type=data_type)
        assert data_layout is expected['spectrum']['data_layout'][dim]


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='...cases.nmrpipe')
def test_nmrpipe_spectrum_convert(expected):
    """Test the NMRPipeSpectrum convert method"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    for dim in tuple(range(spectrum.ndims)) + (-1,):
        # Check points -> Hz
        value1 = spectrum.convert(0, UnitType.POINTS, UnitType.HZ, dim=dim)
        value2 = spectrum.convert(1, UnitType.POINTS, UnitType.HZ, dim=dim)
        value3 = spectrum.convert(spectrum.npts[dim] - 1, UnitType.POINTS,
                                  UnitType.HZ, dim=dim)
        assert value1 - value3 == pytest.approx(spectrum.sw_hz[dim])
        assert value1 - value2 == pytest.approx(spectrum.sw_hz[dim] /
                                                spectrum.npts[dim])

        # Check Hz -> points
        value1 = spectrum.convert(value1, UnitType.HZ, UnitType.POINTS, dim=dim)
        value2 = spectrum.convert(value2, UnitType.HZ, UnitType.POINTS, dim=dim)
        value3 = spectrum.convert(value3, UnitType.HZ, UnitType.POINTS, dim=dim)
        assert value1 == 0
        assert value2 == 1
        assert value3 == spectrum.npts[dim] - 1

        # Check points -> ppm
        value1 = spectrum.convert(0, UnitType.POINTS, UnitType.PPM, dim=dim)
        value2 = spectrum.convert(1, UnitType.POINTS, UnitType.PPM, dim=dim)
        value3 = spectrum.convert(spectrum.npts[dim] - 1, UnitType.POINTS,
                                  UnitType.PPM, dim=dim)
        assert value1 - value3 == pytest.approx(spectrum.sw_ppm[dim])
        assert value1 - value2 == pytest.approx(spectrum.sw_ppm[dim] /
                                                spectrum.npts[dim])

        # Check ppm -> points
        value1 = spectrum.convert(value1, UnitType.PPM, UnitType.POINTS,
                                  dim=dim)
        value2 = spectrum.convert(value2, UnitType.PPM, UnitType.POINTS,
                                  dim=dim)
        value3 = spectrum.convert(value3, UnitType.PPM, UnitType.POINTS,
                                  dim=dim)
        assert value1 == 0
        assert value2 == 1
        assert value3 == spectrum.npts[dim] - 1


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='...cases.nmrpipe')
def test_nmrpipe_spectrum_array_hz(expected):
    """Test the NMRPipeSpectrum array_hz method"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Set the freq range type to a default freq range with endpoint.
    # [sw/2, -sw/2]
    spectrum.time_range_type = RangeType.FREQ | RangeType.ENDPOINT

    # Check that the frequency series respect the spectral width (within 2
    # decimal places)
    t1 = tuple(f_rng[0] - f_rng[-1] for f_rng in spectrum.array_hz)
    t2 = spectrum.sw_hz
    match_tuple_floats(t1, t2, abs_tol=0.01)


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='...cases.nmrpipe')
def test_nmrpipe_spectrum_array_ppm(expected):
    """Test the NMRPipeSpectrum array_ppm method"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Set the freq range type to a default freq range with endpoint.
    # [sw/2, -sw/2]
    spectrum.time_range_type = RangeType.FREQ | RangeType.ENDPOINT

    # Check that the frequency series respect the spectral width (within 4
    # decimal places)
    t1 = tuple(f_rng[0] - f_rng[-1] for f_rng in spectrum.array_ppm)
    t2 = spectrum.sw_ppm
    match_tuple_floats(t1, t2, abs_tol=0.0001)


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='...cases.nmrpipe')
def test_nmrpipe_spectrum_array_s(expected):
    """Test the NMRPipeSpectrum array_s method"""
    # Load the spectrum
    print(f"Loading spectrum '{expected['filepath']}")
    spectrum = NMRPipeSpectrum(expected['filepath'])

    # Set the time range type to a default time range [0, tmax[
    spectrum.time_range_type = RangeType.TIME

    # Check that the time series respect the spectral widths
    t1 = tuple((t_rng[1] - t_rng[0]) ** -1 for t_rng in spectrum.array_s)
    t2 = spectrum.sw_hz

    # Collect other values used for tests below on group delay
    t0 = tuple(t_rng[0] for t_rng in spectrum.array_s)
    tmax = tuple(t_rng[-1] for t_rng in spectrum.array_s)
    dw = tuple(t_rng[1] - t_rng[0] for t_rng in spectrum.array_s)

    # Check the starting point
    assert all(a[0] == 0.0 for a in spectrum.array_s)

    # Match the range values to within 2 decimals--i.e. 8392.123 and 8392.12
    # match
    match_tuple_floats(t1, t2, abs_tol=0.01)

    # Set the time range type to a default time range with a group delay,
    # This will only apply to the current (last) dimension
    # [-group_delay, tmax - group_delay[
    spectrum.time_range_type = RangeType.TIME | RangeType.GROUP_DELAY

    # Get the group delay value in seconds
    grp_delay = floor(spectrum.group_delay) * dw[-1]

    # Check the first point of all dims except last (current) dimension
    assert all(a[0] == 0.0
               for a in spectrum.array_s[:-1])  # all dims except last

    # Check the first point of the last (current) dimension
    if spectrum.correct_digital_filter:
        # Digital filter correction needed
        assert spectrum.array_s[-1][0] == pytest.approx(t0[-1] - grp_delay)
    else:
        # No digital filter correction needed. Group delay not applied
        assert spectrum.array_s[-1][0] == 0.0

    # Check the last point of all dims except last (current) dimension
    assert all(a[-1] == pytest.approx(tmax)
               for a, tmax in zip(spectrum.array_s[:-1], tmax))

    # Check the last point of the last (current) dimension
    if spectrum.correct_digital_filter:
        # Digital filter correction needed
        assert spectrum.array_s[-1][-1] == pytest.approx(tmax[-1] - grp_delay)
    else:
        # No digital filter correction needed. Group delay not applied
        assert spectrum.array_s[-1][-1] == pytest.approx(tmax[-1])

    # Check that the time series respect the spectral widths
    t1 = tuple((t_rng[1] - t_rng[0]) ** -1 for t_rng in spectrum.array_s)
    t2 = spectrum.sw_hz

    # Match the range values to within 1 decimals--i.e. 8392.13 and 8392.12
    # match
    match_tuple_floats(t1, t2, abs_tol=0.1)


# I/O methods

# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected', parametrize_casesets(
    ('*nmrpipe_complex_spectrum_1d', '*nmrpipe_complex_fid_2d',
     '*nmrpipe_real_spectrum_singlefile_3d'), cases='...cases.nmrpipe',
    prefix='data_'))
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

    # Configure the range types to match NMRPipe's processing
    spectrum.freq_range_type = RangeType.FREQ
    spectrum.time_range_type = RangeType.TIME
    spectrum.unit_range_type = RangeType.UNIT

    # Check the attributes
    match_attributes(spectrum, expected)


# Mutators/Processing methods
# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected, expected_em',
                         parametrize_casesets('*nmrpipe_complex_fid_1d',
                                              '*nmrpipe_complex_fid_em_1d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_'))
def test_nmrpipe_spectrum_apodization_exp(expected, expected_em):
    """Test the NMRPipeSpectrum apodization_exp method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_em['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_em = NMRPipeSpectrum(expected_em['filepath'])

    # Configure the range types to match NMRPipe's processing
    spectrum.time_range_type = RangeType.TIME

    # Get the apodization parameters
    dim = spectrum_em.order[0]
    code = spectrum_em.meta[f'FDF{dim}APODCODE']
    lb = spectrum_em.meta[f'FDF{dim}APODQ1']

    assert code == 2.0  # apodization code for EM

    # Check that an apodization hasn't been applied yet
    assert all(apod is ApodizationType.NONE for apod in spectrum.apodization)

    # Apodization the original dataset
    spectrum.apodization_exp(lb=lb)

    # Check the header
    match_metas(spectrum.meta, spectrum_em.meta)
    assert spectrum.apodization[0] == ApodizationType.EXPONENTIAL

    # Find the tolerance for float errors
    tol = spectrum.data.real.max() * 0.0001

    # Check the values
    if spectrum.ndims == 1:
        for row, (i, j) in enumerate(zip(spectrum.data, spectrum_em.data)):
            print(f"Row #{row}: {i}, {j}")
            assert isclose(i, j, abs_tol=tol)
    else:
        raise NotImplementedError


# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected, expected_sp',
                         parametrize_casesets('*nmrpipe_complex_fid_1d',
                                              '*nmrpipe_complex_fid_sp_1d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_'))
def test_nmrpipe_spectrum_apodization_sine(expected, expected_sp):
    """Test the NMRPipeSpectrum apodization_size method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_sp['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_sp = NMRPipeSpectrum(expected_sp['filepath'])

    # Configure the range types to match NMRPipe's processing
    spectrum.unit_range_type = RangeType.UNIT

    # Get the apodization parameters
    dim = spectrum_sp.order[0]
    code = spectrum_sp.meta[f'FDF{dim}APODCODE']
    off = spectrum_sp.meta[f'FDF{dim}APODQ1']
    end = spectrum_sp.meta[f'FDF{dim}APODQ2']
    power = spectrum_sp.meta[f'FDF{dim}APODQ3']

    assert code == 1.0  # apodization code for SP

    # Check that an apodization hasn't been applied yet
    assert all(apod is ApodizationType.NONE for apod in spectrum.apodization)

    # Apodization the original dataset
    spectrum.apodization_sine(off=off, end=end, power=power)

    # Check the header
    match_metas(spectrum.meta, spectrum_sp.meta)
    assert spectrum.apodization[0] == ApodizationType.SINEBELL

    # Find the tolerance for float errors
    tol = spectrum.data.real.max() * 0.0002

    # Check the values
    if spectrum.ndims == 1:
        for row, (i, j) in enumerate(zip(spectrum.data, spectrum_sp.data)):
            assert isclose(i, j, abs_tol=tol)
    else:
        raise NotImplementedError


# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected, expected_ft',
                         parametrize_casesets('*nmrpipe_complex_fid_1d',
                                              '*nmrpipe_complex_fid_ft_1d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_'))
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


# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected, expected_ps',
                         parametrize_casesets('*nmrpipe_complex_spectrum_1d',
                                              '*nmrpipe_complex_spectrum_ps_1d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_'))
def test_nmrpipe_spectrum_phase(expected, expected_ps):
    """Test the NMRPipeSpectrum phase method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_ps['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_ps = NMRPipeSpectrum(expected_ps['filepath'])

    # Configure the range types to match NMRPipe's processing
    spectrum.unit_range_type = RangeType.UNIT

    # Get the phase to use
    dim = spectrum_ps.order[-1]
    p0 = spectrum_ps.meta[f'FDF{dim}P0']
    p1 = spectrum_ps.meta[f'FDF{dim}P1']

    # Phase the spectrum
    spectrum.phase(p0=p0, p1=p1, discard_imaginaries=True)

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


# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected, expected_tp',
                         parametrize_casesets('*nmrpipe_complex_fid_2d',
                                              '*nmrpipe_complex_fid_tp_2d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_'))
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
    for attr in ('domain_type', 'data_type', 'sw_hz', 'sw_ppm', 'car_hz',
                 'car_ppm', 'obs_mhz', 'label'):
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


# See cases_nmrpipe_spectrum.py for a listing of test cases
@pytest.mark.parametrize('expected, expected_zf',
                         parametrize_casesets('*nmrpipe_complex_fid_1d',
                                              '*nmrpipe_complex_fid_zf_1d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_') +
                         parametrize_casesets('*nmrpipe_real_fid_1d',
                                              '*nmrpipe_real_fid_zf_1d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_') +
                         parametrize_casesets('*nmrpipe_complex_fid_2d',
                                              '*nmrpipe_complex_fid_zf_2d',
                                              cases='...cases.nmrpipe',
                                              prefix='data_'))
def test_nmrpipe_spectrum_zerofill(expected, expected_zf):
    """Test the NMRPipeSpectrum zerofill method"""
    # Load the spectrum and its transpose
    print(f"Loading spectra: '{expected['filepath']}' and "
          f"'{expected_zf['filepath']}'")
    spectrum = NMRPipeSpectrum(expected['filepath'])
    spectrum_ps = NMRPipeSpectrum(expected_zf['filepath'])

    # Get the phase to use
    dim = spectrum_ps.order[-1]
    new_size = -1 * int(spectrum_ps.meta[f'FDF{dim}ZF'])

    # Phase the spectrum
    spectrum.zerofill(size=new_size)

    # Check the header
    match_metas(spectrum.meta, spectrum_ps.meta)

    # Find a tolerance for matching numbers. The numbers do not exactly match
    # the reference dataset due to rounding errors (presumably)
    if spectrum.data.is_complex():
        tol = (max(abs(spectrum.data.real.max()),
                   abs(spectrum.data.real.min())) * 0.0001)
    else:
        tol = max(abs(spectrum.data.max()), abs(spectrum.data.min())) * 0.0001

    # Check the values, row-by-row
    if spectrum.ndims == 1:
        for count, (i, j) in enumerate(zip(spectrum.data, spectrum_ps.data)):
            assert isclose(i, j, abs_tol=tol)
    else:
        for i, (row1, row2) in enumerate(zip(spectrum.data, spectrum_ps.data)):
            print(f"Row #{i}")
            assert all(isclose(i, j) for i, j in zip(row1, row2))

