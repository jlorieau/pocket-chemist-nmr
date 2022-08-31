import sys
from . import cli

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version('pocketchemist_nmr')
__all__ = ('cli',)
