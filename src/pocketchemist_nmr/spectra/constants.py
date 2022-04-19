"""
Constants and enum types for NMRSpectra
"""
import typing as t
import re
from enum import Enum, Flag, auto

__all__ = ('UnitType', 'DomainType', 'DataType', 'DataLayout',
           'ApodizationType', 'RangeType')


# Enumeration types
class UnitType(Enum):
    """The types of units for values"""
    UNKNOWN = 0  # Unknown units

    POINTS = 100  # Unit of points (number of points)
    PERCENT = 110  # Percentage

    HZ = 200  # Frequency in Hz (s^-1)
    PPM = 210  # Frequency in parts-per-million (relative to Larmor freq)

    SEC = 300  # Time in seconds

    @classmethod
    def from_string(cls, string: str) \
            -> t.Tuple[t.Union[float, int, None], 'UnitType']:
        """Convert a unit string with optional value to a float/int value and a
        unit type.

        Parameters
        ----------
        string
            The string to convert

        Returns
        -------
        value, unit
            The parsed value, if present, or None, if not present, and
            the UnitType

        Examples
        --------
        >>> UnitType.from_string("2.34 sec")
        (2.34, <UnitType.SEC: 300>)
        >>> UnitType.from_string("38.0%")
        (38.0, <UnitType.PERCENT: 110>)
        >>> UnitType.from_string("Hz")
        (None, <UnitType.HZ: 200>)
        >>> UnitType.from_string("-1.32e-3 ppm")
        (-0.00132, <UnitType.PPM: 210>)
        >>> UnitType.from_string("332")
        (332, <UnitType.POINTS: 100>)
        """
        match = re.match(r'(?P<value>\-?[\d\.]+[eE]?[\-\+]?\d*)?'
                         r'\s*'
                         r'(?P<unit>[\w\%]+)?', string)
        if match is None:
            raise ValueError(f"Cannot parse and convert the value '{string}'")
        d = match.groupdict()

        # Parse the value
        value = d['value']
        if value is not None:
            try:
                value = int(value)
            except ValueError:
                value = float(value)

        # Parse the unit
        if d['unit'] is None or d['unit'].lower() in ('pt', 'pts'):
            return value, cls.POINTS
        elif d['unit'].lower() in ('%', 'pct', 'percent'):
            return value, cls.PERCENT
        elif d['unit'].lower() in ('hz', 'hertz', 's^-1'):
            return value, cls.HZ
        elif d['unit'].lower() in ('ppm', 'parts-per-million'):
            return value, cls.PPM
        elif d['unit'].lower() in ('s', 'sec', 'second', 'seconds'):
            return value, cls.SEC
        else:
            return value, cls.UNKNOWN


class DomainType(Enum):
    """The data domain type for a dimension"""
    UNKNOWN = 0  # Unknown domain type
    TIME = 1  # Time domain data (in sec)
    FREQ = 2  # frequency domain data (in Hz)


class DataType(Enum):
    """The type of data values"""
    UNKNOWN = 0  # Unknown data type
    REAL = 1  # Real data
    IMAG = 2  # Imaginary data
    COMPLEX = 3  # Complex data containing real and imag components


class DataLayout(Enum):
    """How the data is layed out in a dimension"""

    CONTIGUOUS = 0  # No interleaving
    SINGLE_INTERLEAVE = 10  # Complex real/imag points adjacent to each other
    BLOCK_INTERLEAVE = 20  # Complex real points followed by imag points


class ApodizationType(Enum):
    """The type of apodization applied to a dimension"""

    NONE = 0  # No apodization
    SINEBELL = 10  # Sine-bell
    EXPONENTIAL = 20  # Exponential (Lorentzian)


class RangeType(Flag):
    """The type of range to use for the x-axis of data."""

    UNIT = auto()  # Series as [0., 1.0] in npts
    TIME = auto()  # Time series as [0, t_max] in npts
    FREQ = auto()  # Frequency series [0, F_max] in pts

    CENTER = auto()  # Series from [-min/2, max/2] with 0 at the center
    INVERT = auto()  # Series starting with largest numbers to smallest.
    # Series includes endpoint. eg [0, 1] vs [0, 1[. This should not be enabled
    # to get a range in which dx is equal to the dwell
    ENDPOINT = auto()

    GROUP_DELAY = auto()  # Account for the group delay in creating the series
    