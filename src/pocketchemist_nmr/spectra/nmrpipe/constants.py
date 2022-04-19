"""
Constants for NMRPipe spectra
"""
from enum import Enum
import typing as t

from ..constants import DomainType, DataType, ApodizationType

__all__ = ('header_size_bytes', 'data_size_bytes', 'Plane2DPhase',
           'SignAdjustment')

#: The size of the header in bytes
header_size_bytes = 2048

#: The number of bytes for each data value (float / 4 bytes / 32-bit)
data_size_bytes = 4


# Enumeration types
class Plane2DPhase(Enum):
    """Values for the 2D plane phase--i.e. the 'FD2DPHASE' value """
    NONE = None  # Data is not a 2D plane
    MAGNITUDE = 0.0  # Magnitude mode data
    TPPI = 1.0  # TPPI (Time Proportional Phase Incrementation)
    STATES = 2.0  # States or States-TPPI
    IMAGE = 3.0  # Image data
    ARRAY = 4.0  # Array data


class SignAdjustment(Enum):
    """Values for the sign adjustment needed for FFT"""
    NONE = None  # No sign adjustment needed
    REAL = 1.0  # Sign alternation of the real component
    COMPLEX = 2.0  # Sign alternation of both real and imaginary components
    NEGATE_IMAG = 16.0  # Negate the imaginary component
    REAL_NEGATE_IMAG = 17.0  # Same as REAL + NEGATE_IMAG
    COMPLEX_NEGATE_IMAG = 18.0  # Same as COMPLEX + NEGATE_IMAG


# Mappings between enum values and NMRPipe header values
mappings = {
    'domain_type': {0.0: DomainType.TIME,
                    1.0: DomainType.FREQ,
                    None: DomainType.UNKNOWN},
    'data_type': {0.0: DataType.COMPLEX,
                  1.0: DataType.REAL,
                  None: DataType.UNKNOWN},
    'apodization': {0.0: ApodizationType.NONE,
                    1.0: ApodizationType.SINEBELL,
                    2.0: ApodizationType.EXPONENTIAL},
    'sign_adjustment': {1.0: SignAdjustment.REAL,
                        2.0: SignAdjustment.COMPLEX,
                        16.0: SignAdjustment.NEGATE_IMAG,
                        17.0: SignAdjustment.REAL_NEGATE_IMAG,
                        18.0: SignAdjustment.COMPLEX_NEGATE_IMAG,
                        0.0: SignAdjustment.NONE},
    'plane2dphase': {0.0: Plane2DPhase.MAGNITUDE,
                     1.0: Plane2DPhase.TPPI,
                     2.0: Plane2DPhase.STATES,
                     3.0: Plane2DPhase.IMAGE,
                     4.0: Plane2DPhase.ARRAY,
                     None: Plane2DPhase.NONE},
}


def find_mapping(name, cnst, reverse=False, round_cnst=True) \
        -> t.Union[float, DomainType, SignAdjustment, Plane2DPhase, DataType,
                   ApodizationType]:
    """Find the mapping for constant (enum) values.

    Parameters
    ----------
    name
        The name of the mapping to use. e.g. 'domain_type'
    cnst
        The cnst to retrieve for the mapping
    reverse
        If False (default), map the cnst to the mapping dict key
        If True, map the cnst to the mapping dict value
    round_cnst
        If the cnst is a floating point number, round it to the nearest
        integer float value. (ex: 3.9 -> 4.0)
    """
    d_mapping = mappings[name]
    if reverse:
        d_mapping = {v: k for k, v in d_mapping.items()}

    # Clean the cnst, if needed
    if round_cnst and isinstance(cnst, float):
        cnst = round(cnst, 1)
    return d_mapping[cnst]
