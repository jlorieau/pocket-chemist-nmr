"""
Processors for saving and loading spectra
"""
import typing as t

from pocketchemist.utils.types import FilePaths
from pocketchemist.utils.list import wraplist

from .processor import NMRProcessor
from ..spectra import NMRSpectrum, NMRPipeSpectrum


class LoadSpectra(NMRProcessor):
    """Load (one or more) NMR spectra"""

    required_params = ('in_filepaths', 'format')

    def process(self,
                spectra: t.Optional[t.List[NMRSpectrum]] = None,
                in_filepaths: t.Optional[FilePaths] = None,
                **kwargs: t.Any):
        """Load or iterate spectra into the kwargs

        Parameters
        ----------
        spectra
            The spectra to process
        in_filepaths
            The paths for NMR spectra files to load
        """
        # Setup the arguments
        spectra = wraplist(spectra)

        # Load the spectra, if they haven't been loaded yet
        if len(spectra) == 0:
            # Load the filepaths for the spectra
            in_filepaths = wraplist(in_filepaths,
                                    default=wraplist(self.in_filepaths))

            for in_filepath in in_filepaths:
                if self.format.lower() == 'nmrpipe':
                    spectra.append(NMRPipeSpectrum(in_filepath=in_filepath))
                else:
                    raise NotImplementedError(f"An NMR spectrum of format "
                                              f"'{self.format}' is not "
                                              f"supported")

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
        out_filepaths = wraplist(out_filepaths,
                                 default=wraplist(self.out_filepaths))

        # Save the spectra
        for spectrum, out_filepath in zip(spectra, out_filepaths):
            spectrum.save(out_filepath=out_filepath, format=format,
                          overwrite=overwrite)

        # Place the resulting spectra in kwargs and return
        kwargs['spectra'] = spectra
        return kwargs
