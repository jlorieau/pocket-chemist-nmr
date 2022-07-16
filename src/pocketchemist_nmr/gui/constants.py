from enum import Enum


class Tool(Enum):
    """The tools available for working with spectral plots"""

    #: The default mode to pan and zoom
    NAVIGATION = 'Navigation'

    #: Add peaks mode
    ADDPEAKS = 'Add Peaks'

    #: Horizontal 1D trace mode
    HTRACE = "Horizontal 1D"

    #: Vertical 1D trace mode
    VTRACE = "Vertical 1D"
