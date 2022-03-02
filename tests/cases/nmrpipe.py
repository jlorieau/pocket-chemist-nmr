"""
Test cases for spectral data
"""
from pathlib import Path

from pytest_cases import case
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
            'sw_hz': (10000.,),  # Spectral width in Hz
            'sw_ppm': (20.005120939771814,),  # Spectral width in ppm
            'car_hz': (2385.8889820554177,),  # Carrier frequency in Hz
            'car_ppm': (4.7729997634887695,),  # Carrier frequency in Hz
            'range_hz': ((7384.53391599118,
                          -2602.950439453125),),  # Freq ranges in Hz
            'range_ppm': ((14.772849407325031,
                           -5.2072338341491955),),  # Freq ranges in ppm
            'obs_mhz': (499.872009,),  # Observe frequency in MHz
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
def data_nmrpipe_real_fid_1d():
    """A real 1d Free-Induction Decay (FID)"""
    d = data_nmrpipe_complex_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_real.fid')
    d['header']['data_type'] = (DataType.REAL,)  # Type of data
    d['header']['data_pts'] = (799 * 1,)
    d['spectrum']['data_type'] = (DataType.REAL,)
    d['spectrum']['data_layout'] = (DataLayout.CONTIGUOUS,)
    d['spectrum']['data_heights'] = (((0,), 0. + 0.j),
                                     ((-1,), -359985.70000))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_fid_em_1d():
    d = data_nmrpipe_complex_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_em.fid')
    d['spectrum']['apodization'] = (ApodizationType.EXPONENTIAL,)
    d['spectrum']['data_heights'] = (((0,), 0. + 0.j),
                                     ((-1,), -29343.56000 - 1338.36200j))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_fid_sp_1d():
    """A complex 1d Free-Induction Decay (FID) with SP apodization"""
    d = data_nmrpipe_complex_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_sp.fid')
    d['spectrum']['apodization'] = (ApodizationType.SINEBELL,)
    d['spectrum']['data_heights'] = (((0,), 0. + 0.j),
                                     ((-1,), -22273.27000 - 1015.88500j))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_fid_ft_1d():
    """A complex 1d Free-Induction Decay (FID) after automatic Fourier
    transformation"""
    d = data_nmrpipe_complex_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_ft.fid')
    d['spectrum']['domain_type'] = (DomainType.FREQ,)
    d['spectrum']['correct_digital_filter'] = False
    d['spectrum']['data_heights'] = (((0,), 889948.60000 - 789496.60000j),
                                     ((-1,), 1106565.00000 - 778211.60000j))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_fid_zf_1d():
    """A complex 1d Free-Induction Decay (FID) after zero-filling"""
    d = data_nmrpipe_complex_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_zf.fid')
    d['header']['data_pts'] = (799 * 2 * 2,)
    d['header']['pts'] = (799 * 2,)
    d['spectrum']['shape'] = (799 * 1 * 2,)
    d['spectrum']['range_hz'] = ((7385.888906237777, -2607.853271484375),)
    d['spectrum']['range_ppm'] = ((14.775560081700569, -5.21704200892245),)
    d['spectrum']['data_heights'] = (((0,), 0 + 0j),
                                     ((-1,), 0 + 0j))
    return d


@case(tags='singlefile')
def data_nmrpipe_real_fid_zf_1d():
    """A real 1d Free-Induction Decay (FID) after zero-filling"""
    d = data_nmrpipe_real_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' / 'spec_real_zf.fid')
    d['header']['data_type'] = (DataType.REAL,)
    d['header']['data_pts'] = (799 * 1 * 2,)
    d['header']['pts'] = (799 * 1 * 2,)
    d['spectrum']['range_hz'] = ((7385.888906237777, -2607.853271484375),)
    d['spectrum']['range_ppm'] = ((14.775560081700569, -5.21704200892245),)
    d['spectrum']['shape'] = (799 * 1 * 2,)
    d['spectrum']['data_heights'] = (((0,), 0 + 0j),
                                     ((-1,), 0 + 0j))
    return d


@case(tags='singlefile')
def data_nmrpipe_real_spectrum_1d():
    """A real 1d spectrum after SP apodization, zero-filling (ZF), Fourier
    transformation"""
    d = data_nmrpipe_real_fid_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' /
                     'oneone-echo_N-dcpl.jll.ft')
    d['header']['data_type'] = (DataType.REAL,)
    d['header']['data_pts'] = (8192 * 1,)
    d['header']['pts'] = (8192,)
    d['spectrum']['shape'] = (8192 * 1,)
    d['spectrum']['domain_type'] = (DomainType.FREQ,)
    d['spectrum']['data_type'] = (DataType.REAL,)
    d['spectrum']['data_layout'] = (DataLayout.CONTIGUOUS,)
    d['spectrum']['range_hz'] = ((7385.888916015625, -2612.890380859375),)
    d['spectrum']['range_ppm'] = ((14.775560101261272, -5.227118807145823),)
    d['spectrum']['apodization'] = (ApodizationType.SINEBELL,)
    d['spectrum']['correct_digital_filter'] = False
    d['spectrum']['data_heights'] = (((0,), 491585.80000),
                                     ((-1,), 594718.70000))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_spectrum_1d():
    """A complex 1d spectrum after SP apodization, zero-filling (ZF), Fourier
    transformation"""
    d = data_nmrpipe_real_spectrum_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' /
                     'oneone-echo_N-dcpl.jll_complex.ft')
    d['header']['data_type'] = (DataType.COMPLEX,)
    d['header']['data_pts'] = (8192 * 2,)
    d['header']['pts'] = (8192,)
    d['spectrum']['data_type'] = (DataType.COMPLEX,)
    d['spectrum']['data_layout'] = (DataLayout.BLOCK_INTERLEAVE,)
    d['spectrum']['data_heights'] = (((0,), 491585.80000 - 1010224.00000j),
                                     ((-1,), 594718.70000 - 968423.10000j))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_spectrum_ps_1d():
    """A complex 1d spectrum after SP apodization, zero-filling (ZF), Fourier
    transformation, and phasing (PS)"""
    d = data_nmrpipe_real_spectrum_1d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_100_ubq_oneone1d' /
                     'oneone-echo_N-dcpl.jll_ps.ft')
    d['spectrum']['data_heights'] = (((0,), -137928.90000),
                                     ((-1,), -18755.06000))
    return d


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
            'sw_hz': (1671.682007, 8012.820801),  # Spectral width in Hz
            'sw_ppm': (33.00001890141511, 16.029744918834815),
            'car_hz': (5955.541133651248, 2385.8889820554177),  # Carrier freq
            'car_ppm': (117.56600189208984, 4.7729997634887695),
            'obs_mhz': (50.65700149536133, 499.87200927734375),  # Observe freq
            'range_hz': ((6791.381935, 5128.785156),
                         (6392.299548, -1608.001221)),  # Freq ranges in Hz
            'range_ppm': ((134.0660073496, 101.2453363),
                          (12.78787255, -3.2168258891)),  # Freq ranges in ppm
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
    d = data_nmrpipe_complex_fid_2d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_tp.fid')

    # Reverse the following header and spectrum entries
    for entry in ('order', 'data_pts', 'pts'):
        d['header'][entry] = d['header'][entry][::-1]

    for entry in ('order', 'shape', 'sw_hz', 'sw_ppm', 'car_hz', 'car_ppm',
                  'range_hz', 'range_ppm', 'obs_mhz', 'label'):
        d['spectrum'][entry] = d['spectrum'][entry][::-1]

    d['spectrum']['shape'] = (640 * 2, 184 * 1)
    d['spectrum']['data_heights'] = (((0, 0), 0. + 0.j),
                                     ((0, -1), 0. + 0.j),
                                     ((1, 0), 0. + 0.j),
                                     ((-1, -1), 290.00000 - 510.00000j))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_fid_zf_2d():
    """A complex 2d Free-Induction Decay (FID) after zero-filling"""
    d = data_nmrpipe_complex_fid_2d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_zf.fid')
    d['header']['data_pts'] = (640 * 4, 184 * 2)
    d['header']['pts'] = (640 * 2, 184)
    d['spectrum']['shape'] = (184 * 2, 640 * 2)
    d['spectrum']['range_hz'] = ((6791.38193479, 5128.78516),
                                 (6392.29930992, -1614.26147461))
    d['spectrum']['range_ppm'] = ((134.0660073, 101.2453363),
                                  (12.787872078, -3.2293496028))
    d['spectrum']['data_heights'] = (((0, 0), 0. + 0.j),
                                     ((0, -1), 0. + 0.j),
                                     ((1, 0), 0. + 0.j),
                                     ((-1, -1), 0. + 0.j))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_fid_tp_zf_2d():
    """A complex 2d Free-Induction Decay (FID) with transpose (TP) and
    zero-filling"""
    d = data_nmrpipe_complex_fid_tp_2d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_tp_zf.fid')
    d['header']['data_pts'] = (184 * 4, 640 * 2)
    d['header']['pts'] = (184 * 2, 640)
    d['spectrum']['shape'] = (640 * 2, 184 * 2)
    d['spectrum']['range_hz'] = ((6392.29954758, -1608.00122070),
                                 (6791.38206847, 5124.2426758))
    d['spectrum']['range_ppm'] = ((12.7878725533, -3.216825889147),
                                  (134.0660099886, 101.1556650516))
    d['spectrum']['data_heights'] = (((0, 0), 0. + 0.j),
                                     ((0, -1), 0. + 0.j),
                                     ((1, 0), 0. + 0.j),
                                     ((-1, -1), 0. + 0.j))
    return d


@case(tags='singlefile')
def data_nmrpipe_real_spectrum_2d():
    """A real 2d spectrum after solvent suppression (SOL), SP apodization,
    zero-filling (ZF), Fourier transformation, phasing, transposition and
    region extraction (EXT)"""
    d = data_nmrpipe_complex_fid_tp_2d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                     'hsqcetfpf3gpsi2.ft2')
    d['header']['data_type'] = (DataType.REAL, DataType.REAL)
    d['header']['data_pts'] = (368 * 1, 1024 * 1)
    d['header']['pts'] = (368, 1024)
    d['spectrum']['shape'] = (1024 * 1, 368 * 1)
    d['spectrum']['domain_type'] = (DomainType.FREQ, DomainType.FREQ)
    d['spectrum']['data_type'] = (DataType.REAL, DataType.REAL)
    d['spectrum']['data_layout'] = (DataLayout.CONTIGUOUS,
                                    DataLayout.CONTIGUOUS,)
    d['spectrum']['sw_hz'] = (2003.205200, 1671.682007)
    d['spectrum']['sw_ppm'] = (4.007436229708704, 33.00001890141511)
    d['spectrum']['range_hz'] = ((5126.11335322, 3123.39721680),
                                 (6791.38206847, 5124.2426758))
    d['spectrum']['range_ppm'] = ((10.25485175822, 6.24839390650),
                                  (134.0660099886, 101.1556650516))
    d['spectrum']['apodization'] = (ApodizationType.SINEBELL,
                                    ApodizationType.SINEBELL)
    d['spectrum']['correct_digital_filter'] = False
    d['spectrum']['data_heights'] = (((0, 0), 282333.20000),
                                     ((0, -1), 104637.30000),
                                     ((1, 0), 252427.00000),
                                     ((-1, -1), -110738.90000))
    return d


@case(tags='singlefile')
def data_nmrpipe_complex_spectrum_2d():
    """A complex 2d spectrum after solvent suppression (SOL), SP apodization,
    zero-filling (ZF), Fourier transformation, phasing, transposition and
    region extraction (EXT)"""
    d = data_nmrpipe_complex_fid_2d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                     'hsqcetfpf3gpsi2_complex.ft2')
    d['header']['data_type'] = (DataType.COMPLEX, DataType.COMPLEX)
    d['header']['data_pts'] = (1024 * 2, 368 * 2)
    d['header']['pts'] = (1024, 368)
    d['spectrum']['shape'] = (368 * 2, 1024 * 1)
    d['spectrum']['domain_type'] = (DomainType.FREQ, DomainType.FREQ)
    d['spectrum']['data_type'] = (DataType.COMPLEX, DataType.COMPLEX)
    d['spectrum']['sw_hz'] = (1671.682007, 2003.205200)
    d['spectrum']['sw_ppm'] = (33.00001890141511, 4.007436229708704)
    d['spectrum']['range_hz'] = ((6791.38206847, 5124.2426758),
                                 (5126.11335322, 3123.3972168))
    d['spectrum']['range_ppm'] = ((134.0660099886, 101.1556650516),
                                  (10.25485175822, 6.2483939065))
    d['spectrum']['apodization'] = (ApodizationType.SINEBELL,
                                    ApodizationType.SINEBELL)
    d['spectrum']['correct_digital_filter'] = False
    d['spectrum']['data_heights'] = (((0, 0), 282333.20000 - 39091.02000j),
                                     ((0, -1), 37383.56000 + 228056.20000j),
                                     ((1, 0), -242140.50000 + 213173.60000j),
                                     ((-1, -1), -69947.75000 - 507403.40000j))
    return d


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
           'ndims': 2,
           'order': (1, 2),
           'shape': (39 * 2, 559),
           'domain_type': (DomainType.TIME, DomainType.TIME),
           'data_type': (DataType.COMPLEX, DataType.COMPLEX,),
           'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                           DataLayout.BLOCK_INTERLEAVE,),
           'sw_hz': (1671.682007, 6996.26904296875),
           'sw_ppm': (33.000018901415, 13.9961208331771),
           'car_hz': (5955.490504476, 2385.8889820554),
           'car_ppm': (117.56500244140, 4.7729997634888),
           'obs_mhz': (50.65700149536, 499.87200927734),
           'range_hz': ((6769.89990860, 5141.081543),
                        (5882.43963524, -1101.31372070)),
           'range_ppm': ((133.6419390955, 101.4880745249),
                         (11.76789163240, -2.20319141753)),
           'label': ('15N', 'HN'),
           'apodization': (ApodizationType.NONE, ApodizationType.NONE),
           'group_delay': 67.98582458496094,
           'correct_digital_filter': True,
           'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
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
            'shape': (51 * 2, 39 * 2, 559),
            'domain_type': (DomainType.TIME, DomainType.TIME,
                            DomainType.TIME),
            'data_type': (DataType.COMPLEX, DataType.COMPLEX,
                          DataType.COMPLEX,),
            'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.SINGLE_INTERLEAVE,
                            DataLayout.BLOCK_INTERLEAVE,),
            'sw_hz': (1445.921997, 1671.682007, 6996.26904296875),
            'sw_ppm': (11.5016786743931, 33.000018901415, 13.9961208331771),
            'car_hz': (22244.210891840, 5955.490504476, 2385.8889820554),
            'car_ppm': (176.9429931640, 117.56500244140, 4.7729997634888),
            'obs_mhz': (125.71399688720, 50.65700149536, 499.87200927734),
            'range_hz': ((22952.9963666, 21535.4258),
                         (6769.89990860, 5141.081543),
                         (5882.43963524, -1101.31372070)),
            'range_ppm': ((182.58107239, 171.304916831),
                          (133.6419390955, 101.4880745249),
                          (11.76789163240, -2.20319141753)),
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
    d = data_nmrpipe_complex_fid_3d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'ft2' /
                     'spec%03d.ft2')
    d['header']['order'] = (2, 1, 3)
    d['header']['data_type'] = (DataType.REAL, DataType.REAL, DataType.COMPLEX)
    d['header']['data_pts'] = (220 * 1, 256 * 1, 51 * 2)
    d['header']['pts'] = (220, 256, 51)
    d['spectrum']['order'] = (3, 1, 2)
    d['spectrum']['shape'] = (51 * 2, 256 * 1, 220 * 1)
    d['spectrum']['domain_type'] = (DomainType.TIME, DomainType.FREQ,
                                    DomainType.FREQ)
    d['spectrum']['data_type'] = (DataType.COMPLEX, DataType.REAL,
                                  DataType.REAL)
    d['spectrum']['data_layout'] = (DataLayout.SINGLE_INTERLEAVE,
                                    DataLayout.CONTIGUOUS,
                                    DataLayout.CONTIGUOUS,)
    d['spectrum']['sw_hz'] = (1445.921997070, 1671.682006836, 2753.451172)
    d['spectrum']['sw_ppm'] = (11.50167867439, 33.0000189014, 5.508312369512)
    d['spectrum']['range_hz'] = ((22952.9963666, 21535.425781),
                                 (6791.3316865, 5126.1796875),
                                 (5752.3572860, 3003.83178711))
    d['spectrum']['range_ppm'] = ((182.58107239, 171.304916831),
                                  (134.065015418, 101.1939028403),
                                  (11.50766031962, 6.00920181839))
    d['spectrum']['apodization'] = (ApodizationType.NONE,
                                    ApodizationType.SINEBELL,
                                    ApodizationType.NONE)
    d['spectrum']['correct_digital_filter'] = False
    d['spectrum']['data_heights'] = (((0, 0, 0), -114428.60000),
                                     ((0, 0, -1), 43824.73000),
                                     ((0, 1, 0), -137487.00000),
                                     ((0, -1, 0), -47025.36000),
                                     ((1, 0, 0), 91014.32000),
                                     ((-1, 0, 0), -3568.61700),
                                     ((-1, -1, -1), 1224.21000))
    return d


@case(tags='singlefile')
def data_nmrpipe_real_spectrum_singlefile_3d():
    """A real 3d spectrum (single file) after solvent suppression (SOL),
    SP apodization, zero-filling (ZF), Fourier transformation, phasing,
    transposition and region extraction (EXT)"""
    d = data_nmrpipe_rrc_spectrum_3d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'hncogp3d.ft3')
    d['header']['order'] = (2, 3, 1)
    d['header']['data_type'] = (DataType.REAL, DataType.REAL, DataType.REAL)
    d['header']['data_pts'] = (220 * 1, 512 * 1, 256 * 1)
    d['header']['pts'] = (220, 512, 256)
    d['spectrum']['order'] = (1, 3, 2)
    d['spectrum']['shape'] = (256 * 1, 512 * 1, 220 * 1)
    d['spectrum']['domain_type'] = (DomainType.FREQ, DomainType.FREQ,
                                    DomainType.FREQ)
    d['spectrum']['data_type'] = (DataType.REAL, DataType.REAL, DataType.REAL)
    d['spectrum']['data_layout'] = (DataLayout.CONTIGUOUS,
                                    DataLayout.CONTIGUOUS,
                                    DataLayout.CONTIGUOUS)
    d['spectrum']['sw_hz'] = (1671.682007, 1445.921997, 2753.451171875)
    d['spectrum']['sw_ppm'] = (33.0000189014, 11.50167867439, 5.508312369512)
    d['spectrum']['car_hz'] = (5955.49050448, 22244.21089184, 2385.888982055)
    d['spectrum']['car_ppm'] = (117.5650024414, 176.94299316, 4.772999763489)
    d['spectrum']['obs_mhz'] = (50.6570014954, 125.7139968872, 499.8720092773)
    d['spectrum']['range_hz'] = ((6791.3316865, 5126.1796875),
                                 (22967.17214942, 21524.07422),
                                 (5752.3572860, 3003.83178710))
    d['spectrum']['range_ppm'] = ((134.065015418, 101.1939028403),
                                  (182.6938345618, 171.214620104),
                                  (11.50766031962, 6.00920181839))
    d['spectrum']['label'] = ('15N', '13C', 'HN')
    d['spectrum']['apodization'] = (ApodizationType.SINEBELL,
                                    ApodizationType.SINEBELL,
                                    ApodizationType.NONE)
    d['spectrum']['data_heights'] = (((0, 0, 0), -1199905.00000),
                                     ((0, 0, -1), -2773523.00000),
                                     ((0, 1, 0), -1257113.00000),
                                     ((0, -1, 0), -1079424.00000),
                                     ((1, 0, 0), -1271509.00000),
                                     ((-1, 0, 0), -872241.80000),
                                     ((-1, -1, -1), -2219091.00000))
    return d


@case(tags='multifile')
def data_nmrpipe_real_spectrum_3d():
    """A real 3d spectrum (2d planes) after solvent suppression (SOL),
    SP apodization, zero-filling (ZF), Fourier transformation, phasing,
    transposition and region extraction (EXT)"""
    d = data_nmrpipe_real_spectrum_singlefile_3d()
    d['filepath'] = (Path('data') / 'bruker' /
                     'CD20170124_av500hd_103_ubq_hnco3d' / 'ft3' /
                     'spec%04d.ft3')
    return d
