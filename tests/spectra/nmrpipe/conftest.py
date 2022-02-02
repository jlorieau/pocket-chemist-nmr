"""Utilities, functions and fixtures for NMRPipeSpectrum tests"""
from pathlib import Path
from pocketchemist_nmr.spectra.constants import DataType, DomainType
from pocketchemist_nmr.spectra.nmrpipe.constants import SignAdjustment


def expected(multifile: bool = None) -> tuple[dict, ...]:
    """Return an iteratable of dicts with expected values

    Parameters
    ----------
    multifile
        If True, only return multifile datasets
        If False, do not return multifile datasets
        If None, return datasets regardless of whether they're multifile
    """
    spectra = (
        # 1D spectra
        {'title': '1d complex fid',
         'filepath': Path('data') / 'bruker' /
          'CD20170124_av500hd_100_ubq_oneone1d' / 'spec.fid',
         'ndims': 1,  # Number of dimensions in spectrum
         'order': (2,),  # Order of data
         'data_type': (DataType.COMPLEX,),  # Type of data
         'data_pts': (799 * 2,),
         'pts': (799,),
         'shape': (799 * 1,),
         'domain_type': (DomainType.TIME,),  # Type of domain for each dim
         'sw': (10000.,),  # The spectra width in Hz
         'label': ('1H',),  # The labels for each dimension
         'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
         'data_heights': (((0,), 0. + 0.j),
                          ((-1,), -359985.70000 - 16418.97000j))},

        {'title': '1d real fid',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_100_ubq_oneone1d' / 'oneone-echo_N-dcpl.jll.ft',
         'ndims': 1,
         'order': (2,),
         'data_type': (DataType.REAL,),
         'data_pts': (8192 * 1,),
         'pts': (8192,),
         'shape': (8192 * 1,),
         'domain_type': (DomainType.FREQ,),
         'sw': (10000.,),
         'label': ('1H',),
         'sign_adjustment': (SignAdjustment.NONE,),
         'data_heights': (((0,), 491585.80000),
                          ((-1,), 594718.70000))},

        {'title': '1d real spectrum',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_100_ubq_oneone1d' /
         'oneone-echo_N-dcpl.jll_complex.ft',
         'ndims': 1,
         'order': (2,),
         'data_type': (DataType.COMPLEX,),
         'data_pts': (8192 * 2,),
         'pts': (8192,),
         'shape': (8192 * 1,),
         'domain_type': (DomainType.FREQ,),
         'sw': (10000.,),
         'label': ('1H',),
         'sign_adjustment': (SignAdjustment.NONE,),
         'data_heights': (((0,), 491585.80000 - 1010224.00000j),
                          ((-1,), 594718.70000 - 968423.10000j)),},

        # 2D spectra
        {'title': '2d complex fid',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec.fid',
         'ndims': 2,  # Number of dimensions in spectrum
         # Data ordering of data. (direct, indirect) e.g. F1, F2
         'order': (2, 1),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX),  # Type of data
         'data_pts': (640 * 2, 184 * 2),  # Num of real + imag pts, data ordered
         'pts': (640, 184),  # Num of complex or real pts, data ordered
         # Shape of returned tensor (indirect, direct), reverse of pts
         'shape': (184 * 2, 640 * 1),
         'domain_type': (DomainType.TIME, DomainType.TIME),
         'sw': (8012.820801, 1671.682007),
         'label': ('HN', '15N'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
         'data_heights': (((0, 0), 0. + 0.j),
                          ((0, -1), 5619.12600 - 4132.80200j),
                          ((1, 0), 0. + 0.j),
                          ((-1, -1), -761.71680 - 996.09120j))},

        {'title': '2d real spectrum',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'hsqcetfpf3gpsi2.ft2',
         'ndims': 2,
         'order': (1, 2),
         'data_type': (DataType.REAL, DataType.REAL),
         'data_pts': (368 * 1, 1024 * 1),
         'pts': (368, 1024),
         'shape': (1024 * 1, 368 * 1),
         'domain_type': (DomainType.FREQ, DomainType.FREQ),
         'sw': (1671.682007, 2003.205200),
         'label': ('15N', 'HN'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
         'data_heights': (((0, 0), 551430.50000),
                          ((0, -1), 204368.90000),
                          ((1, 0), 493020.70000),
                          ((-1, -1), -216286.20000))},

        {'title': '2d complex spectrum',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'hsqcetfpf3gpsi2_complex.ft2',
         'ndims': 2, 'order': (2, 1),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX),
         'data_pts': (1024 * 2, 368 * 2),
         'pts': (1024, 368),
         'shape': (368 * 2, 1024 * 1),
         'domain_type': (DomainType.FREQ, DomainType.FREQ),
         'sw': (2003.205200, 1671.682007),
         'label': ('HN', '15N'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
         'data_heights': (((0, 0), 551430.50000 - 76349.69000j),
                          ((0, -1), 73015.44000 + 445419.70000j),
                          ((1, 0), -472929.90000 + 416353.10000j),
                          ((-1, -1), -136615.50000 - 991015.50000j))},

        # 3D Spectra
        {'title': '3d complex fid (2d plane)',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_103_ubq_hnco3d' / 'fid' / 'spec001.fid',
         'ndims': 3,
         'order': (2, 1, 3),
         'data_type': (DataType.COMPLEX, DataType.COMPLEX, DataType.COMPLEX,),
         'data_pts': (559 * 2, 39 * 2, 51 * 2),
         'pts': (559, 39, 51),
         'shape': (39 * 2, 559),
         'domain_type': (DomainType.TIME, DomainType.TIME, DomainType.TIME),
         'sw': (6996.269043, 1671.682007, 1445.921997),
         'label': ('HN', '15N', '13C'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                             SignAdjustment.NONE),
         'data_heights': (((0, 0), 0. + 0.j),
                          ((0, -1), -6837.88700 + 5389.64600j),
                          ((1, 0), 0. + 0.j),
                          ((-1, -1), -676.75750 + 13228.51000j,))},

        {'title': '3d real spectrum',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_103_ubq_hnco3d' / 'hncogp3d.ft3',
         'ndims': 3, 'order': (2, 3, 1),
         'data_type': (DataType.REAL, DataType.REAL, DataType.REAL,),
         'data_pts': (220 * 1, 512 * 1, 256 * 1),
         'pts': (220, 512, 256),
         'shape': (256, 512, 220),
         'domain_type': (DomainType.FREQ, DomainType.FREQ, DomainType.FREQ),
         'sw': (6996.269043, 1445.921997, 1671.682007),
         'label': ('HN', '13C', '15N'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                             SignAdjustment.NONE),
         'data_heights': (((0, 0, 0), -1199905.00000),
                          ((0, 0, -1), -2773523.00000),
                          ((0, 1, 0), -1257113.00000),
                          ((0, -1, 0), -1079424.00000),
                          ((1, 0, 0), -1271509.00000),
                          ((-1, 0, 0), -872241.80000),
                          ((-1, -1, -1), -2219091.00000))},

        {'title': '3d real/real/complex spectrum (2d planes)',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_103_ubq_hnco3d' / 'ft2' / 'spec%03d.ft2',
         'multifile': True,
         'ndims': 3, 'order': (2, 1, 3),
         'data_type': (DataType.REAL, DataType.REAL, DataType.COMPLEX),
         'data_pts': (220 * 1, 256 * 1, 51 * 2),
         'pts': (220, 256, 51),
         'shape': (51 * 2, 256 * 1, 220 * 1),
         'domain_type': (DomainType.FREQ, DomainType.FREQ, DomainType.FREQ),
         'sw': (6996.269043, 1445.921997, 1671.682007),
         'label': ('HN', '13C', '15N'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                             SignAdjustment.NONE),
         'data_heights': (((0, 0, 0), -114428.60000),
                          ((0, 0, -1), 43824.73000),
                          ((0, 1, 0), -137487.00000),
                          ((0, -1, 0), -47025.36000),
                          ((1, 0, 0), 91014.32000),
                          ((-1, 0, 0), -3568.61700),
                          ((-1, -1, -1), 1224.21000))},

        {'title': '3d real spectrum (2d planes)',
         'filepath': Path('data') / 'bruker' /
         'CD20170124_av500hd_103_ubq_hnco3d' / 'ft3' / 'spec%04d.ft3',
         'multifile': True,
         'ndims': 3, 'order': (2, 3, 1),
         'data_type': (DataType.REAL, DataType.REAL, DataType.REAL,),
         'data_pts': (220 * 1, 512 * 1, 256 * 1),
         'pts': (220, 512, 256),
         'shape': (256, 512, 220),
         'domain_type': (DomainType.FREQ, DomainType.FREQ, DomainType.FREQ),
         'sw': (6996.269043, 1445.921997, 1671.682007),
         'label': ('HN', '13C', '15N'),
         'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                             SignAdjustment.NONE),
         'data_heights': (((0, 0, 0), -1199905.00000),
                          ((0, 0, -1), -2773523.00000),
                          ((0, 1, 0), -1257113.00000),
                          ((0, -1, 0), -1079424.00000),
                          ((1, 0, 0), -1271509.00000),
                          ((-1, 0, 0), -872241.80000),
                          ((-1, -1, -1), -2219091.00000))},
    )

    # Filter spectra
    if multifile is True:
        spectra = tuple(spectrum for spectrum in spectra
                        if spectrum.get('multifile', False))
    elif multifile is False:
        spectra = tuple(spectrum for spectrum in spectra
                        if not spectrum.get('multifile', False))
    else:
        pass

    return spectra
