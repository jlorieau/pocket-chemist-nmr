from enum import Enum


class MouseMode(Enum):
    """The mouse navigation and selection states."""

    #: The default mode to pan and zoom
    NAVIGATION = 'Navigation'

    #: Add peaks mode
    ADDPEAKS = 'Add Peaks'
