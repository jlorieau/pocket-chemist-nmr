"""
Test cases for test_nmrpipe_spectrum.py
"""
from pytest_cases import get_all_cases, parametrize


def get_data_case(glob, cases='...cases.nmrpipe', prefix='data_'):
    """Retrieve a data test case with the given suffix name"""
    return get_all_cases(lambda: None, cases=cases, prefix=prefix,
                         glob=glob)[0]()  # eval the function


@parametrize('glob', ('*nmrpipe_complex_spectrum_1d',
                      '*nmrpipe_complex_fid_2d',
                      '*nmrpipe_real_spectrum_singlefile_3d'))
def case_nmrpipe_io(glob):
    """A series of test cases for testing IO (saving, loading) functions"""
    return get_data_case(glob)


def case_nmrpipe_compare_em_1d():
    """Compare exponential multiply between reference and comparison 1Ds"""
    return (get_data_case('*nmrpipe_complex_fid_1d'),
            get_data_case('*nmrpipe_complex_fid_em_1d'))


def case_nmrpipe_compare_sp_1d():
    """Compare sinebell power between reference and comparison 1Ds"""
    return (get_data_case('*nmrpipe_complex_fid_1d'),
            get_data_case('*nmrpipe_complex_fid_sp_1d'))


def case_nmrpipe_compare_ft_1d():
    """Compare Fourier transform between reference and comparison 1Ds"""
    return (get_data_case('*nmrpipe_complex_fid_1d'),
            get_data_case('*nmrpipe_complex_fid_ft_1d'))


def case_nmrpipe_compare_ps_1d():
    """Compare phasing between reference and comparison 1Ds"""
    return (get_data_case('*nmrpipe_complex_spectrum_1d'),
            get_data_case('*nmrpipe_complex_spectrum_ps_1d'))


def case_nmrpipe_compare_tp_2d():
    """Compare transpose between reference and comparison 2Ds"""
    return (get_data_case('*nmrpipe_complex_fid_2d'),
            get_data_case('*nmrpipe_complex_fid_tp_2d'))


def case_nmrpipe_compare_zf_1d():
    """Compare zero-fill between reference and comparison 2Ds"""
    return (get_data_case('*nmrpipe_complex_fid_1d'),
            get_data_case('*nmrpipe_complex_fid_zf_1d'))
