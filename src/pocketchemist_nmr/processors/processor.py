"""
Processors for NMR spectra
"""
import typing as t

import scipy.fft as fft
from pocketchemist.processors import Processor, GroupProcessor

from ..spectra import NMRPipeSpectrum

__all__ = ('LoadSpectra',)


class NMRProcessor(Processor):
    """A processing step for NMR spectra"""


class NMRGroupProcessor(GroupProcessor):
    """A group processor for NMR spectra"""

    def process(self, **kwargs):
        for processor in self.processors:
            kwargs = processor.process(**kwargs)
        return kwargs


class LoadSpectra(NMRProcessor):
    """Load an NMR spectra (one or more)"""

    required_params = ('in_filepath', 'format')

    def process(self, in_filepath: t.Union[str, 'pathlib.Path'] = None,
                **kwargs):
        """Load the spectrum into the kwargs"""
        # Setup the arguments
        in_filepath = (in_filepath if in_filepath is not None else
                       self.in_filepath)

        if self.format.lower() == 'nmrpipe':
            kwargs['spectra'] = NMRPipeSpectrum(in_filepath=in_filepath)
        else:
            raise NotImplementedError(f"An NMR spectrum of format "
                                      f"'{self.format}' is not supported")
        return kwargs


class FTSpectra(NMRProcessor):
    """Fourier Transform spectra (one or more)"""

    #: Fourier Transform mode
    #: - 'auto': determine which method to use based on the spectra
    #: - 'inv': Fourier Transform a frequency spectrum to a time-domain
    #: - 'real': Real Fourier Transform
    required_params = ('mode',)

    def process(self, spectra, mode=None, **kwargs):
        mode = mode if mode is not None else self.mode

        if mode == 'auto':
            # Perform the fft
            fft.fft(spectra.data)
        else:
            raise NotImplementedError(f"Class '{self.__class__.__name__}' does"
                                      f"not support mode '{mode}'")

        # Setup the arguments that are passed to future processors
        kwargs['spectra'] = spectra
        return kwargs
