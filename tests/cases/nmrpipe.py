"""
Test cases for spectral data
"""
from pathlib import Path

from pytest_cases import parametrize, case
from pocketchemist_nmr.spectra.constants import (DataType, DomainType,
                                                 ApodizationType, DataLayout)
from pocketchemist_nmr.spectra.nmrpipe.constants import (SignAdjustment,
                                                         Plane2DPhase)


@case(tags='singlefile')
def data_nmrpipe_complex_fid_1d():
    """A complex 1d Free-Induction Decay (FID)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec.fid'),
        # Header (meta values). NMRPipe ordering (inner-outer1-outer2)
        'header': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_pts': (799 * 2,),
            'pts': (799,)},
        # Spectra accessor values. Torch ordering (outer2-outer1-inner)
        'spectrum': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'shape': (799 * 1,),
            'domain_type': (DomainType.TIME,),  # Each dim's domain
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_layout': (DataLayout.BLOCK_INTERLEAVE,),  # Data layout
            'sw': (10000.,),  # The spectra width in Hz
            'label': ('1H',),  # The labels for each dimension
            'apodization': (ApodizationType.NONE,),  # Apodization type
            # Digital filter group delay
            'group_delay': 67.98625183105469,
            # Whether a digital filter correction is needed
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
            'plane2dphase': Plane2DPhase.MAGNITUDE,  # Type of 2d phase
            'data_heights': (((0,), 0. + 0.j),
                             ((-1,), -359985.70000 - 16418.97000j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_em_1d():
    """A complex 1d Free-Induction Decay (FID) with EM apodization"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_em.fid'),
        # Header (meta values). NMRPipe ordering (inner-outer1-outer2)
        'header': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_pts': (799 * 2,),
            'pts': (799,)},
        # Spectra accessor values. Torch ordering (outer2-outer1-inner)
        'spectrum': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'shape': (799 * 1,),
            'domain_type': (DomainType.TIME,),  # Each dim's domain
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_layout': (DataLayout.BLOCK_INTERLEAVE,),  # Data layout
            'sw': (10000.,),  # The spectra width in Hz
            'label': ('1H',),  # The labels for each dimension
            'apodization': (ApodizationType.EXPONENTIAL,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
            'plane2dphase': Plane2DPhase.MAGNITUDE,  # Type of 2d phase
            'data_heights': (((0,), 0. + 0.j),
                             ((-1,), -29343.56000 - 1338.36200j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_sp_1d():
    """A complex 1d Free-Induction Decay (FID) with SP apodization"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_sp.fid'),
        'header': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_pts': (799 * 2,),
            'pts': (799,)},
        # Spectra accessor values. Torch ordering (outer2-outer1-inner)
        'spectrum': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'shape': (799 * 1,),
            'domain_type': (DomainType.TIME,),  # Each dim's domain
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_layout': (DataLayout.BLOCK_INTERLEAVE,),  # Data layout
            'sw': (10000.,),  # The spectra width in Hz
            'label': ('1H',),  # The labels for each dimension
            'apodization': (ApodizationType.SINEBELL,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
            'plane2dphase': Plane2DPhase.MAGNITUDE,  # Type of 2d phase
            'data_heights': (((0,), 0. + 0.j),
                             ((-1,), -22273.27000 - 1015.88500j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_ft_1d():
    """A complex 1d Free-Induction Decay (FID) after automatic Fourier
    transformation"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_ft.fid'),
        # Header (meta values). NMRPipe ordering (inner-outer1-outer2)
        'header': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_pts': (799 * 2,),
            'pts': (799,)},
        # Spectra accessor values. Torch ordering (outer2-outer1-inner)
        'spectrum': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'shape': (799 * 1,),
            'domain_type': (DomainType.FREQ,),  # Each dim's domain
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_layout': (DataLayout.BLOCK_INTERLEAVE,),  # Data layout
            'sw': (10000.,),  # The spectra width in Hz
            'label': ('1H',),  # The labels for each dimension
            'apodization': (ApodizationType.NONE,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
            'plane2dphase': Plane2DPhase.MAGNITUDE,  # Type of 2d phase
            'data_heights': (((0,), 889948.60000 - 789496.60000j),
                             ((-1,), 1106565.00000 - 778211.60000j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_zf_1d():
    """A complex 1d Free-Induction Decay (FID) after zero-filling"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_zf.fid'),
        # Header (meta values). NMRPipe ordering (inner-outer1-outer2)
        'header': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_pts': (799 * 2 * 2,),
            'pts': (799 * 2,)},
        # Spectra accessor values. Torch ordering (outer2-outer1-inner)
        'spectrum': {
            'ndims': 1,  # Number of dimensions in spectrum
            'order': (2,),  # Order of data
            'shape': (799 * 1 * 2,),
            'domain_type': (DomainType.TIME,),  # Each dim's domain
            'data_type': (DataType.COMPLEX,),  # Type of data
            'data_layout': (DataLayout.BLOCK_INTERLEAVE,),  # Data layout
            'sw': (10000.,),  # The spectra width in Hz
            'label': ('1H',),  # The labels for each dimension
            'apodization': (ApodizationType.NONE,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
            'plane2dphase': Plane2DPhase.MAGNITUDE,  # Type of 2d phase
            'data_heights': (((0,), 0 + 0j),
                             ((-1,), 0 + 0j))},
    }


@case(tags='singlefile')
def data_nmrpipe_real_spectrum_1d():
    """A real 1d spectrum after SP apodization, zero-filling (ZF), Fourier
    transformation"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' /
                     'oneone-echo_N-dcpl.jll.ft'),
        'header': {
            'ndims': 1,
            'order': (2,),
            'data_type': (DataType.REAL,),
            'data_pts': (8192 * 1,),
            'pts': (8192,)},
        'spectrum': {
            'ndims': 1,
            'order': (2,),
            'shape': (8192 * 1,),
            'domain_type': (DomainType.FREQ,),
            'data_type': (DataType.REAL,),
            'data_layout': (DataLayout.CONTIGUOUS,),
            'sw': (10000.,),
            'label': ('1H',),
            'apodization': (ApodizationType.SINEBELL,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE,),
            'plane2dphase': Plane2DPhase.MAGNITUDE,
            'data_heights': (((0,), 491585.80000),
                             ((-1,), 594718.70000))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_spectrum_1d():
    """A complex 1d spectrum after SP apodization, zero-filling (ZF), Fourier
    transformation"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' /
                     'oneone-echo_N-dcpl.jll_complex.ft'),
        'header': {
            'ndims': 1,
            'order': (2,),
            'data_type': (DataType.COMPLEX,),
            'data_pts': (8192 * 2,),
            'pts': (8192,)},
        'spectrum': {
            'ndims': 1,
            'order': (2,),
            'shape': (8192 * 1,),
            'domain_type': (DomainType.FREQ,),
            'data_type': (DataType.COMPLEX,),
            'data_layout': (DataLayout.BLOCK_INTERLEAVE,),
            'sw': (10000.,),
            'label': ('1H',),
            'apodization': (ApodizationType.SINEBELL,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE,),
            'plane2dphase': Plane2DPhase.MAGNITUDE,
            'data_heights': (((0,), 491585.80000 - 1010224.00000j),
                             ((-1,), 594718.70000 - 968423.10000j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_spectrum_ps_1d():
    """A complex 1d spectrum after SP apodization, zero-filling (ZF), Fourier
    transformation, and phasing (PS)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' /
                     'oneone-echo_N-dcpl.jll_ps.ft'),
        'header': {
            'ndims': 1,
            'order': (2,),
            'data_type': (DataType.REAL,),
            'data_pts': (8192 * 1,),
            'pts': (8192,)},
        'spectrum': {
            'ndims': 1,
            'order': (2,),
            'shape': (8192 * 1,),
            'domain_type': (DomainType.FREQ,),
            'data_type': (DataType.REAL,),
            'data_layout': (DataLayout.CONTIGUOUS,),
            'sw': (10000.,),
            'label': ('1H',),
            'apodization': (ApodizationType.SINEBELL,),
            'group_delay': 67.98625183105469,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE,),
            'plane2dphase': Plane2DPhase.MAGNITUDE,
            'data_heights': (((0,), -137928.90000),
                             ((-1,), -18755.06000))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_2d():
    """A complex 2d Free-Induction Decay (FID)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec.fid'),
        'header': {
            'ndims': 2,  # Number of dimensions in spectrum
            # Data ordering of data. (direct, indirect) e.g. F1, F2
            'order': (2, 1),
            # Type of data (Complex/Real/Imag)
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_pts': (640 * 2, 184 * 2),  # Num of real + imag pts
            'pts': (640, 184)},  # Num of complex or real pts, data ordered
        'spectrum': {
            'ndims': 2,
            'order': (1, 2),
            # Shape of returned tensor (indirect, direct), reverse of pts
            'shape': (184 * 2, 640 * 1),
            'domain_type': (DomainType.TIME, DomainType.TIME),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw': (1671.682007, 8012.820801),
            'label': ('15N', 'HN'),
            'apodization': (ApodizationType.NONE, ApodizationType.NONE),
            'group_delay': 67.98423767089844,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0), 0. + 0.j),
                             ((0, -1), 2877.00000 - 2116.00000j),
                             ((1, 0), 0. + 0.j),
                             ((-1, -1), -390.00000 - 510.00000j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_tp_2d():
    """A complex 2d Free-Induction Decay (FID) with transpose (TP)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_tp.fid'),
        'header': {
            'ndims': 2,  # Number of dimensions in spectrum
            # Data ordering of data. (direct, indirect) e.g. F1, F2
            'order': (1, 2),
            # Type of data (Complex/Real/Imag)
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_pts': (184 * 2, 640 * 2),  # Num of real + imag pts
            'pts': (184, 640)},  # Num of complex or real pts, data ordered
        'spectrum': {
            'ndims': 2,
            'order': (2, 1),
            # Shape of returned tensor (indirect, direct), reverse of pts
            'shape': (640 * 2, 184 * 1),
            'domain_type': (DomainType.TIME, DomainType.TIME),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw': (8012.820801, 1671.682007),
            'label': ('HN', '15N'),
            'apodization': (ApodizationType.NONE, ApodizationType.NONE),
            'group_delay': 67.98423767089844,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0), 0. + 0.j),
                             ((0, -1), 0. + 0.j),
                             ((1, 0), 0. + 0.j),
                             ((-1, -1), 290.00000 - 510.00000j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_zf_2d():
    """A complex 2d Free-Induction Decay (FID) after zero-filling"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_zf.fid'),
        'header': {
            'ndims': 2,  # Number of dimensions in spectrum
            # Data ordering of data. (direct, indirect) e.g. F1, F2
            'order': (2, 1),
            # Type of data (Complex/Real/Imag)
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_pts': (640 * 4, 184 * 2),  # Num of real + imag pts
            'pts': (640 * 2, 184)},  # Num of complex or real pts, data ordered
        'spectrum': {
            'ndims': 2,
            'order': (1, 2),
            # Shape of returned tensor (indirect, direct), reverse of pts
            'shape': (184 * 2, 640 * 2),
            'domain_type': (DomainType.TIME, DomainType.TIME),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw': (1671.682007, 8012.820801),
            'label': ('15N', 'HN'),
            'apodization': (ApodizationType.NONE, ApodizationType.NONE),
            'group_delay': 67.98423767089844,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0), 0. + 0.j),
                             ((0, -1), 0. + 0.j),
                             ((1, 0), 0. + 0.j),
                             ((-1, -1), 0. + 0.j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_tp_zf_2d():
    """A complex 2d Free-Induction Decay (FID) with transpose (TP) and
    zero-filling"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_tp_zf.fid'),
        'header': {
            'ndims': 2,  # Number of dimensions in spectrum
            # Data ordering of data. (direct, indirect) e.g. F1, F2
            'order': (1, 2),
            # Type of data (Complex/Real/Imag)
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_pts': (184 * 4, 640 * 2),  # Num of real + imag pts
            'pts': (184 * 2, 640)},  # Num of complex or real pts, data ordered
        'spectrum': {
            'ndims': 2,
            'order': (2, 1),
            # Shape of returned tensor (indirect, direct), reverse of pts
            'shape': (640 * 2, 184 * 2),
            'domain_type': (DomainType.TIME, DomainType.TIME),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw': (8012.820801, 1671.682007),
            'label': ('HN', '15N'),
            'apodization': (ApodizationType.NONE, ApodizationType.NONE),
            'group_delay': 67.98423767089844,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0), 0. + 0.j),
                             ((0, -1), 0. + 0.j),
                             ((1, 0), 0. + 0.j),
                             ((-1, -1), 0. + 0.j))},
    }

@case(tags='singlefile')
def data_nmrpipe_real_spectrum_2d():
    """A real 2d spectrum after solvent suppression (SOL), SP apodization,
    zero-filling (ZF), Fourier transformation, phasing, transposition and
    region extraction (EXT)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                     'hsqcetfpf3gpsi2.ft2'),
        'header': {
            'ndims': 2,
            'order': (1, 2),
            'data_type': (DataType.REAL, DataType.REAL),
            'data_pts': (368 * 1, 1024 * 1),
            'pts': (368, 1024)},
        'spectrum': {
            'ndims': 2,
            'order': (2, 1),
            'shape': (1024 * 1, 368 * 1),
            'domain_type': (DomainType.FREQ, DomainType.FREQ),
            'data_type': (DataType.REAL, DataType.REAL),
            'data_layout': (DataLayout.CONTIGUOUS,
                            DataLayout.CONTIGUOUS,),
            'sw': (2003.205200, 1671.682007),
            'label': ('HN', '15N'),
            'apodization': (ApodizationType.SINEBELL,
                            ApodizationType.SINEBELL),
            'group_delay': 67.98423767089844,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0), 282333.20000),
                             ((0, -1), 104637.30000),
                             ((1, 0), 252427.00000),
                             ((-1, -1), -110738.90000))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_spectrum_2d():
    """A complex 2d spectrum after solvent suppression (SOL), SP apodization,
    zero-filling (ZF), Fourier transformation, phasing, transposition and
    region extraction (EXT)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                     'hsqcetfpf3gpsi2_complex.ft2'),
        'header': {
            'ndims': 2,
            'order': (2, 1),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_pts': (1024 * 2, 368 * 2),
            'pts': (1024, 368)},
        'spectrum': {
            'ndims': 2,
            'order': (1, 2),
            'shape': (368 * 2, 1024 * 1),
            'domain_type': (DomainType.FREQ, DomainType.FREQ),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw': (1671.682007, 2003.205200),
            'label': ('15N', 'HN'),
            'apodization': (ApodizationType.SINEBELL,
                            ApodizationType.SINEBELL),
            'group_delay': 67.98423767089844,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0), 282333.20000 - 39091.02000j),
                             ((0, -1), 37383.56000 + 228056.20000j),
                             ((1, 0), -242140.50000 + 213173.60000j),
                             ((-1, -1), -69947.75000 - 507403.40000j))},
    }


@case(tags='singlefile')
def data_nmrpipe_complex_fid_plane_3d():
    """A complex 2d plane of a 3d FID"""
    return {
       'filepath': (Path('data') / 'bruker' /
                    'CD20170124_av500hd_103_ubq_hnco3d' / 'fid' /
                    'spec001.fid'),
       'header': {
           'ndims': 3,
           'order': (2, 1, 3),
           'data_type': (DataType.COMPLEX, DataType.COMPLEX,
                         DataType.COMPLEX,),
           'data_pts': (559 * 2, 39 * 2, 51 * 2),
           'pts': (559, 39, 51)},
       'spectrum': {
           'ndims': 3,
           'order': (3, 1, 2),
           'shape': (39 * 2, 559),
           'domain_type': (DomainType.TIME, DomainType.TIME,
                           DomainType.TIME),
           'data_type': (DataType.COMPLEX, DataType.COMPLEX,
                         DataType.COMPLEX,),
           'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                           DataLayout.SINGLE_INTERLEAVE,
                           DataLayout.BLOCK_INTERLEAVE,),
           'sw': (1445.921997, 1671.682007, 6996.26904296875),
           'label': ('13C',  '15N', 'HN'),
           'apodization': (ApodizationType.NONE, ApodizationType.NONE,
                           ApodizationType.NONE),
           'group_delay': 67.98582458496094,
           'correct_digital_filter': True,
           'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                               SignAdjustment.NONE),
           'plane2dphase': Plane2DPhase.STATES,
           'data_heights': (((0, 0), 0. + 0.j),
                            ((0, -1), -6837.88700 + 5389.64600j),
                            ((1, 0), 0. + 0.j),
                            ((-1, -1), -676.75750 + 13228.51000j,))},
   }


@case(tags='multifile')
def data_nmrpipe_complex_fid_3d():
    """A complex 3d FID split over multiple 2d planes"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'fid' /
                     'spec%03d.fid'),
        'header': {
            'ndims': 3,
            'order': (2, 1, 3),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX,
                          DataType.COMPLEX,),
            'data_pts': (559 * 2, 39 * 2, 51 * 2),
            'pts': (559, 39, 51)},
        'spectrum': {
            'ndims': 3,
            'order': (3, 1, 2),
            'shape': (51*2, 39 * 2, 559),
            'domain_type': (DomainType.TIME, DomainType.TIME,
                            DomainType.TIME),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX,
                          DataType.COMPLEX,),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw': (1445.921997, 1671.682007, 6996.26904296875),
            'label': ('13C',  '15N', 'HN'),
            'apodization': (ApodizationType.NONE, ApodizationType.NONE,
                            ApodizationType.NONE),
            'group_delay': 67.98582458496094,
            'correct_digital_filter': True,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                                SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0, 0), 0. + 0.j),
                             ((0, 0, -1), -6837.88700 + 5389.64600j),
                             ((0, 1, 0), 0. + 0.j),
                             ((0, -1, 0), 0. + 0.j),
                             ((1, 0, 0), 0. + 0.j),
                             ((-1, 0, 0), 0. + 0.j),
                             ((-1, -1, -1), 1723.63200 -2121.09300j))},
    }


@case(tags='multifile')
def data_nmrpipe_rrc_spectrum_3d():
    """A partially transformed 3D spectrum with Real/Real/Complex data after
    solvent suppression (SOL), SP apodization, zero-filling (ZF), Fourier
    transformation, phasing, transposition and region extraction (EXT)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'ft2' /
                     'spec%03d.ft2'),
        'header': {
            'ndims': 3,
            'order': (2, 1, 3),
            'data_type': (DataType.REAL, DataType.REAL, DataType.COMPLEX),
            'data_pts': (220 * 1, 256 * 1, 51 * 2),
            'pts': (220, 256, 51)},
        'spectrum': {
            'ndims': 3,
            'order': (3, 1, 2),
            'shape': (51 * 2, 256 * 1, 220 * 1),
            'domain_type': (DomainType.TIME, DomainType.FREQ,
                            DomainType.FREQ),
            'data_type': (DataType.COMPLEX, DataType.REAL, DataType.REAL),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.CONTIGUOUS,
                            DataLayout.CONTIGUOUS,),
            'sw': (1445.921997, 1671.682007, 2753.451171875),
            'label': ('13C', '15N', 'HN'),
            'apodization': (ApodizationType.NONE,
                            ApodizationType.SINEBELL,
                            ApodizationType.NONE),
            'group_delay': 67.98582458496094,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                                SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0, 0), -114428.60000),
                             ((0, 0, -1), 43824.73000),
                             ((0, 1, 0), -137487.00000),
                             ((0, -1, 0), -47025.36000),
                             ((1, 0, 0), 91014.32000),
                             ((-1, 0, 0), -3568.61700),
                             ((-1, -1, -1), 1224.21000))},
    }


@case(tags='singlefile')
def data_nmrpipe_real_spectrum_singlefile_3d():
    """A real 3d spectrum (single file) after solvent suppression (SOL),
    SP apodization, zero-filling (ZF), Fourier transformation, phasing,
    transposition and region extraction (EXT)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'hncogp3d.ft3'),
        'header': {
            'ndims': 3,
            'order': (2, 3, 1),
            'data_type': (DataType.REAL, DataType.REAL, DataType.REAL,),
            'data_pts': (220 * 1, 512 * 1, 256 * 1),
            'pts': (220, 512, 256)},
        'spectrum': {
            'ndims': 3,
            'order': (1, 3, 2),
            'shape': (256, 512, 220),
            'domain_type': (DomainType.FREQ, DomainType.FREQ,
                            DomainType.FREQ),
            'data_type': (DataType.REAL, DataType.REAL, DataType.REAL,),
            'data_layout': (DataLayout.CONTIGUOUS,
                            DataLayout.CONTIGUOUS,
                            DataLayout.CONTIGUOUS,),
            'sw': (1671.682007, 1445.921997, 2753.451171875),
            'label': ('15N', '13C', 'HN'),
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                                SignAdjustment.NONE),
            'apodization': (ApodizationType.SINEBELL,
                            ApodizationType.SINEBELL,
                            ApodizationType.NONE),
            'group_delay': 67.98582458496094,
            'correct_digital_filter': False,
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0, 0), -1199905.00000),
                             ((0, 0, -1), -2773523.00000),
                             ((0, 1, 0), -1257113.00000),
                             ((0, -1, 0), -1079424.00000),
                             ((1, 0, 0), -1271509.00000),
                             ((-1, 0, 0), -872241.80000),
                             ((-1, -1, -1), -2219091.00000))},
    }


@case(tags='multifile')
def data_nmrpipe_real_spectrum_3d():
    """A real 3d spectrum (2d planes) after solvent suppression (SOL),
    SP apodization, zero-filling (ZF), Fourier transformation, phasing,
    transposition and region extraction (EXT)"""
    return {
        'filepath': (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'ft3' /
                     'spec%04d.ft3'),
        'header': {
            'ndims': 3,
            'order': (2, 3, 1),
            'data_type': (DataType.REAL, DataType.REAL, DataType.REAL,),
            'data_pts': (220 * 1, 512 * 1, 256 * 1),
            'pts': (220, 512, 256)},
        'spectrum': {
            'ndims': 3,
            'order': (1, 3, 2),
            'shape': (256, 512, 220),
            'domain_type': (DomainType.FREQ, DomainType.FREQ,
                            DomainType.FREQ),
            'data_type': (DataType.REAL, DataType.REAL, DataType.REAL,),
            'data_layout': (DataLayout.CONTIGUOUS,
                            DataLayout.CONTIGUOUS,
                            DataLayout.CONTIGUOUS,),
            'sw': (1671.682007, 1445.921997, 2753.451171875),
            'label': ('15N', '13C', 'HN',),
            'apodization': (ApodizationType.SINEBELL,
                            ApodizationType.SINEBELL,
                            ApodizationType.NONE),
            'group_delay': 67.98582458496094,
            'correct_digital_filter': False,
            'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                                SignAdjustment.NONE),
            'plane2dphase': Plane2DPhase.STATES,
            'data_heights': (((0, 0, 0), -1199905.00000),
                             ((0, 0, -1), -2773523.00000),
                             ((0, 1, 0), -1257113.00000),
                             ((0, -1, 0), -1079424.00000),
                             ((1, 0, 0), -1271509.00000),
                             ((-1, 0, 0), -872241.80000),
                             ((-1, -1, -1), -2219091.00000))},
    }
