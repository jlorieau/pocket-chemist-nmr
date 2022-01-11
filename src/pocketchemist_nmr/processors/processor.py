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

    required_params = ('in_filepath', 'format')


class FTSpectra(NMRProcessor):
    """Fourier Transform spectra (one or more)"""

    #: Fourier Transform mode
    #: - 'auto': determine which method to use based on the spectra
    #: - 'inv': Fourier Transform a frequency spectrum to a time-domain
    #: - 'real': Real Fourier Transform
    required_params = ('mode',)
