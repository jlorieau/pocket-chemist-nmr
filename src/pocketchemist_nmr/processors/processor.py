"""
Processors for NMR spectra
"""
import typing as t
from itertools import zip_longest

import scipy.fft as fft
from pocketchemist.processors import Processor, GroupProcessor
from pocketchemist.utils.types import FilePaths

from ..spectra import NMRSpectrum, NMRPipeSpectrum

__all__ = ('NMRGroupProcessor', 'LoadSpectra', 'SaveSpectra', 'FTSpectra')


class NMRProcessor(Processor):
    """A processing step for NMR spectra"""


class NMRGroupProcessor(GroupProcessor):
    """A group processor for NMR spectra"""

    def process(self, **kwargs):
        for processor in self.processors:
            kwargs = processor.process(**kwargs)
        return kwargs


class LoadSpectra(NMRProcessor):
    """Load (one or more) NMR spectra"""

    required_params = ('in_filepaths', 'format')

    def process(self, in_filepaths: t.Optional[FilePaths] = None,
                **kwargs):
        """Load spectra into the kwargs

        Parameters
        ----------
        in_filepaths
            The paths for NMR spectra files to load
        """
        # Setup the arguments
        in_filepaths = (in_filepaths if in_filepaths is not None else
                        self.in_filepaths)

        spectra = []
        for in_filepath in in_filepaths:
            if self.format.lower() == 'nmrpipe':
                spectra.append(NMRPipeSpectrum(in_filepath=in_filepath))
            else:
                raise NotImplementedError(f"An NMR spectrum of format "
                                          f"'{self.format}' is not supported")

        kwargs['spectra'] = spectra
        return kwargs


class SaveSpectra(NMRProcessor):
    """Save (one or more) NMR spectra"""

    required_params = ('out_filepaths', 'format')

    def process(self,
                spectra: t.List[NMRSpectrum],
                out_filepaths: t.Optional[FilePaths] = None,
                format: str = None,
                overwrite: bool = True,
                **kwargs):
        """Save spectra into the kwargs

        Parameters
        ----------
        spectra
            A list of :obj:`pocketchemist_nmr.spectra.NMRSpectrum` objects
        out_filepaths
            The paths for NMR spectra files to write
        format
            If specified, save the spectra in the givn format. The default is
            to use the same format as used to load the spectrum
        overwrite
            If True (default), overwrite existing files
        """
        # Setup arguments
        if out_filepaths is not None:
            pass
        elif 'out_filepaths' in self.params:
            out_filepaths = self.params['out_filepaths']
        else:
            out_filepaths = []

        # Save the spectra
        for spectrum, out_filepath in zip_longest(spectra, out_filepaths,
                                                  fillvalue=None):
            spectrum.save(out_filepath=out_filepath, format=format,
                          overwrite=overwrite)


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
