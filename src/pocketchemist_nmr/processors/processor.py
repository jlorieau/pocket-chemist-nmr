"""
Processors for NMR spectra
"""
from pocketchemist.processors import Processor, GroupProcessor

__all__ = ('LoadSpectra',)


class NMRProcessor(Processor):
    """A processing step for an NMR spectrum"""


class LoadSpectra(NMRProcessor):
    """Load an NMR spectra (one or more)"""

    # The loaded NMR spectra
    spectra = None

    required_params = ('in_filepath',)


class FTSpectra(NMRProcessor):
    """Fourier Transform spectra (one or more)"""
    