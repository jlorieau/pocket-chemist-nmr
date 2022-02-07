import importlib.metadata

from . import cli, processors, spectra

__all__ = ('cli', 'processors', 'spectra')

__version__ = importlib.metadata.version("pocketchemist_nmr")
