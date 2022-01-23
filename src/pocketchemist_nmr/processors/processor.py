"""
Processors for NMR spectra
"""
import typing as t
from multiprocessing import Pool

from loguru import logger
from pocketchemist.processors import Processor, GroupProcessor
from pocketchemist.processors.fft import FFTProcessor

from ..spectra import NMRSpectrum

__all__ = ('NMRProcessor', 'NMRGroupProcessor', 'FTSpectra')


def set_logger(logger_):
    """Setup a shared logger for multiprocessing"""
    global logger
    logger = logger_


class NMRProcessor(Processor):
    """A processing step for NMR spectra"""


class NMRGroupProcessor(GroupProcessor):
    """A group processor for NMR spectra"""

    def process(self, **kwargs):
        # Setup a spectra list
        return self.process_pool()
        #return self.process_sequence(**kwargs)

    def process_sequence(self, **kwargs):
        """Process subprocessors in sequence"""
        spectra = []
        for processor in self.processors:
            kwargs = processor.process(**kwargs)
            logger.debug(f"Running {processor.__class__.__name__} "
                         f"with: {kwargs}")
        return kwargs

    def process_pool(self, **kwargs):
        """Process subprocessed with a pool"""
        spectra = []

        # Setup a pool and pass a shared logger
        with Pool(initializer=set_logger, initargs=(logger, )) as pool:
            logger.debug(f"Setting up pool: {pool}")
            results = []
            for i in range(1):
                result = pool.apply_async(NMRGroupProcessor.process_sequence,
                                          (self,), kwds=kwargs)
                results.append(result)

            # Wait for the results to finish
            [result.get() for result in results]



class FTSpectra(FFTProcessor, NMRProcessor):
    """Fourier Transform spectra (one or more)"""

    #: Fourier Transform mode
    #: - 'auto': determine which method to use based on the spectra
    #: - 'inv': Fourier Transform a frequency spectrum to a time-domain
    #: - 'real': Real Fourier Transform
    required_params = ('mode',)

    def process(self,
                spectra: t.Iterable[NMRSpectrum],
                mode: str = None,
                **kwargs):

        # Setup the fft/ifft functions
        ft_func = self.get_module_callable(category='fft')
        ft_opts = {'auto': self.mode == 'auto'}

        # Perform the Fourier transformation
        for spectrum in spectra:
            spectrum.ft(ft_func=ft_func, ft_opts=ft_opts)

        # Setup the arguments that are passed to future processors
        kwargs['spectra'] = spectra
        return kwargs
