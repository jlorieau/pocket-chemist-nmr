"""Utilities, functions and fixtures for NMRPipeSpectrum tests.

See note in :class:`.NMRPipeSpectrum`. The data ordering for NMRPipe files
is in reverse order from torch tensors and the base :class:`.NMRSpectrum`.
"""
from pathlib import Path
import typing as t

from pocketchemist_nmr.spectra.constants import DataType, DomainType, DataLayout
from pocketchemist_nmr.spectra.nmrpipe.constants import (SignAdjustment,
                                                         Plane2DPhase)


def expected(include: t.Optional[t.Tuple[str, ...]] = None,
             multifile: bool = None) -> dict:
    """Return an iteratable of dicts with expected values

    Parameters
    ----------
    include
        If specified, only include spectra whose title (key) matches items
        in this list
    multifile
        If True, only return multifile datasets
        If False, do not return multifile datasets
        If None, return datasets regardless of whether they're multifile
    """
    spectra = {
        # 1D spectra
        '1d complex fid': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_100_ubq_oneone1d' / 'spec.fid'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE,),  # Sign adjustment
                'plane2dphase': Plane2DPhase.MAGNITUDE,  # Type of 2d phase
                'data_heights': (((0,), 0. + 0.j),
                                 ((-1,), -359985.70000 - 16418.97000j))},
            },

        '1d real fid': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_100_ubq_oneone1d' /
                         'oneone-echo_N-dcpl.jll.ft'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE,),
                'plane2dphase': Plane2DPhase.MAGNITUDE,
                'data_heights': (((0,), 491585.80000),
                                 ((-1,), 594718.70000))},
        },

        '1d real spectrum': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_100_ubq_oneone1d' /
                         'oneone-echo_N-dcpl.jll_complex.ft'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE,),
                'plane2dphase': Plane2DPhase.MAGNITUDE,
                'data_heights': (((0,), 491585.80000 - 1010224.00000j),
                                 ((-1,), 594718.70000 - 968423.10000j))},
        },

        # 2D spectra
        '2d complex fid': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec.fid'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0), 0. + 0.j),
                                 ((0, -1), 2877.00000 - 2116.00000j),
                                 ((1, 0), 0. + 0.j),
                                 ((-1, -1), -390.00000 - 510.00000j))},
        },

        '2d complex fid (tp)': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_101_ubq_hsqcsi2d' / 'spec_tp.fid'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0), 0. + 0.j),
                                 ((0, -1), 0. + 0.j),
                                 ((1, 0), 0. + 0.j),
                                 ((-1, -1), 290.00000 - 510.00000j))},
        },

        '2d real spectrum': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                         'hsqcetfpf3gpsi2.ft2'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0), 282333.20000),
                                 ((0, -1), 104637.30000),
                                 ((1, 0), 252427.00000),
                                 ((-1, -1), -110738.90000))},
        },

        '2d complex spectrum': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                         'hsqcetfpf3gpsi2_complex.ft2'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0), 282333.20000 - 39091.02000j),
                                  ((0, -1), 37383.56000 + 228056.20000j),
                                  ((1, 0), -242140.50000 + 213173.60000j),
                                  ((-1, -1), -69947.75000 - 507403.40000j))},
        },

        '2d complex spectrum (tp)': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                         'hsqcetfpf3gpsi2_complex_tp.ft2'),
            'format': 'nmrpipe',
            'header': {
                'ndims': 2,
                'order': (1, 2),
                'data_type': (DataType.COMPLEX, DataType.COMPLEX),
                'data_pts': (368 * 2, 1024 * 2),
                'pts': (368, 1024)},
            'spectrum': {
                'ndims': 2,
                'order': (2, 1),
                'shape': (1024 * 2, 368 * 1),
                'domain_type': (DomainType.FREQ, DomainType.FREQ),
                'data_type': (DataType.COMPLEX, DataType.COMPLEX),
                'data_layout': (DataLayout.SINGLE_INTERLEAVE,
                                DataLayout.BLOCK_INTERLEAVE,),
                'sw': (2003.205200, 1671.682007),
                'label': ('HN', '15N'),
                'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE),
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0), 282333.20000 - 242140.50000j),
                                 ((0, -1), 104637.30000 - 249088.20000j),
                                 ((1, 0), -39091.02000 + 213173.60000j),
                                 ((-1, -1), 148426.70000 - 507403.40000j))},
        },

        # 3D Spectra
        '3d complex fid (2d plane)': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_103_ubq_hnco3d' / 'fid' /
                         'spec001.fid'),
            'format': 'nmrpipe',
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
                'sign_adjustment': (SignAdjustment.NONE, SignAdjustment.NONE,
                                     SignAdjustment.NONE),
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0), 0. + 0.j),
                                  ((0, -1), -6837.88700 + 5389.64600j),
                                  ((1, 0), 0. + 0.j),
                                  ((-1, -1), -676.75750 + 13228.51000j,))},
        },

        '3d complex fid': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_103_ubq_hnco3d' / 'fid' /
                         'spec%03d.fid'),
            'format': 'nmrpipe',
            'header': {
                'multifile': True,
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
        },

        '3d real spectrum': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_103_ubq_hnco3d' / 'hncogp3d.ft3'),
            'format': 'nmrpipe',
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
                'plane2dphase': Plane2DPhase.STATES,
                'data_heights': (((0, 0, 0), -1199905.00000),
                                 ((0, 0, -1), -2773523.00000),
                                 ((0, 1, 0), -1257113.00000),
                                 ((0, -1, 0), -1079424.00000),
                                 ((1, 0, 0), -1271509.00000),
                                 ((-1, 0, 0), -872241.80000),
                                 ((-1, -1, -1), -2219091.00000))},
        },

        '3d real/real/complex spectrum (2d planes)': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_103_ubq_hnco3d' / 'ft2' /
                         'spec%03d.ft2'),
            'format': 'nmrpipe',
            'header': {
                'multifile': True,
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
        },

        '3d real spectrum (2d planes)': {
            'filepath': (Path('data') / 'bruker' /
                         'CD20170124_av500hd_103_ubq_hnco3d' / 'ft3' /
                         'spec%04d.ft3'),
            'format': 'nmrpipe',
            'header': {
                'multifile': True,
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
        },
    }

    # Filter spectra
    if include is not None:
        spectra = {k: v for k, v in spectra.items() if k in include}

    if multifile is True:
        spectra = {k: v for k, v in spectra.items()
                   if v['header'].get('multifile', False)}
    elif multifile is False:
        spectra = {k: v for k, v in spectra.items()
                   if not v['header'].get('multifile', False)}
    else:
        pass

    return spectra
