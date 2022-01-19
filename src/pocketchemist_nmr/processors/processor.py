"""
Processors for NMR spectra
"""
import scipy.fft as fft
from pocketchemist.processors import Processor, GroupProcessor

__all__ = ('NMRProcessor', 'NMRGroupProcessor', 'FTSpectra')


class NMRProcessor(Processor):
    """A processing step for NMR spectra"""


class NMRGroupProcessor(GroupProcessor):
    """A group processor for NMR spectra"""

    def process(self, **kwargs):
        return self.process_sequence(**kwargs)

    def process_sequence(self, **kwargs):
        """Process on sequences of subprocessors """
        for processor in self.processors:
            kwargs = processor.process(**kwargs)
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

        for spectrum in spectra:
            data = spectrum.data
            scale = 1.0 / float(data.shape[-1])

            if mode == 'auto':
                # Perform the fft
                data = (fft.fft(fft.ifftshift(data, -1),
                                axis=-1).astype(data.dtype) * scale)
            else:
                raise NotImplementedError(f"Class '{self.__class__.__name__}' "
                                          f"does not support mode '{mode}'")
            spectrum.data = data

        # Setup the arguments that are passed to future processors
        kwargs['spectra'] = spectra
        return kwargs
