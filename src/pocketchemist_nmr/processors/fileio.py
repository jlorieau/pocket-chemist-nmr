"""
Processors for saving and loading spectra
"""
import typing as t
from itertools import zip_longest

from pocketchemist.utils.types import FilePaths

from .processor import NMRProcessor
from ..spectra import NMRSpectrum, NMRPipeSpectrum


class LoadSpectra(NMRProcessor):
    """Load (one or more) NMR spectra"""

    required_params = ('in_filepaths', 'format')

    def process(self,
                spectra: t.Optional[t.List[NMRSpectrum]] = None,
                in_filepaths: t.Optional[FilePaths] = None,
                **kwargs):
        """Load or iterate spectra into the kwargs

        Parameters
        ----------
        in_filepaths
            The paths for NMR spectra files to load
        """
        # Setup the arguments
        spectra = spectra if spectra is not None else []

        # Load the spectra, if they haven't been loaded yet
        if len(spectra) == 0:
            # Load the filepaths for the spectra
            in_filepaths = (in_filepaths if in_filepaths is not None else
                            self.in_filepaths)

            for in_filepath in in_filepaths:
                if self.format.lower() == 'nmrpipe':
                    spectra.append(NMRPipeSpectrum(in_filepath=in_filepath))
                else:
                    raise NotImplementedError(f"An NMR spectrum of format "
                                              f"'{self.format}' is not "
                                              f"supported")

        # Iterate spectra, if needed
        for spectrum in spectra:
            # Do not iterate if there is not iterator
            if spectrum.iterator is None:
                continue

            # Try the iteration
            try:
                next(spectrum)
            except StopIteration:
                pass

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
