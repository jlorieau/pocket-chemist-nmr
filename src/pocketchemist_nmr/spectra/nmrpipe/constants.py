"""
Constants for NMRPipe spectra
"""
from enum import Enum

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
