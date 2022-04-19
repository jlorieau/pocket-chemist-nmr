"""
Processors for NMR spectra
"""
import typing as t
from multiprocessing import Pool

from loguru import logger
from pocketchemist.processors import Processor, GroupProcessor
from pocketchemist.processors.fft import FFTProcessor

from ..spectra import NMRSpectrum

__all__ = ('NMRProcessor', 'NMRGroupProcessor', 'ApodizationExpSpectra',
           'ApodizationSinebellSpectra', 'ExtractSpectra', 'FTSpectra',
           'PhaseSpectra', 'Transpose2D', 'ZeroFillSpectra')


def set_logger(logger_):
    """Setup a shared logger for multiprocessing"""
    global logger
    logger = logger_


class NMRProcessor(Processor):
    """A processing step for NMR spectra"""

    method = None

    def process(self, spectra: t.Iterable[NMRSpectrum], **kwargs):
        for spectrum in spectra:
            if self.method is None:
                continue

            # Get the method to run and run with the required_params/
            # optional_params
            meth = getattr(spectrum, self.method)
            req_params = {k: getattr(self, k) for k in self.required_params}
            opt_params = {k: getattr(self, k) for k in self.optional_params}
            opt_params.update(req_params)
            meth(**opt_params)

        # Setup the arguments that are passed to future processors
        kwargs['spectra'] = spectra
        return kwargs


class NMRGroupProcessor(GroupProcessor):
    """A group processor for NMR spectra"""

    def process(self, **kwargs):
        # Setup a spectra list
        return self.process_sequence(**kwargs)

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


class ApodizationExpSpectra(NMRProcessor):
    """Apodization with exponential multiply (Lorentzian) in the last dimension
    """
    method = 'apodization_exp'
    required_params = ('lb',)
    optional_params = ('start', 'size')


class ApodizationSinebellSpectra(NMRProcessor):
    """Apodization with sinebell power (SP) in the last dimension
    """
    method = 'apodization_sine'
    required_params = ('off', 'end', 'power')
    optional_params = ('start', 'size')


class ExtractSpectra(NMRProcessor):
    """Extract region in the last dimension
    """
    method = 'extract'
    required_params = ('start', 'unit_start', 'end', 'unit_end')
    optional_params = ('update_meta',)


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

        # Perform the Fourier transformation
        for spectrum in spectra:
            spectrum.ft()

        # Setup the arguments that are passed to future processors
        kwargs['spectra'] = spectra
        return kwargs


class PhaseSpectra(NMRProcessor):
    """Phase the last dimension of a dataset"""
    method = 'phase'
    optional_params = ('p0', 'p1', 'discard_imaginaries')


class Transpose2D(NMRProcessor):
    """Transpose the last 2 dimension (outer1-inner) of a dataset"""

    def process(self, spectra: t.Iterable[NMRSpectrum], **kwargs):
        for spectrum in spectra:
            assert spectrum.ndims > 1, (
                f"Spectrum has {spectrum.ndims} dimensions and at least 2 "
                f"dimensions are required for a transpose."
            )
            # Switch the last 2 dimensions
            order = list(range(spectrum.ndims))  # 0, 1, 2
            spectrum.transpose(dim0=order[-2], dim1=order[-1])
        # Setup the arguments that are passed to future processors
        kwargs['spectra'] = spectra
        return kwargs


class ZeroFillSpectra(NMRProcessor):
    """Zero-fill the last dimension of a dataset"""
    method = 'zerofill'
    optional_params = ('double', 'double_base2', 'size', 'pad')
