"""
NMR Spectra in different formats
"""
import abc
import typing as t
from pathlib import Path
import re

import nmrglue as ng


class NMRSpectrum(abc.ABC):
    """An NMR spectrum base class."""

    #: metadata on the spectrum
    meta: dict

    #: The data for the spectrum, either an array or an iterator
    data: t.Union[t.Iterable, 'numpy.ndarray']

    #: If True, the data attribute is an iterator that must be processed one
    #: at a time.
    is_data_iterator: bool

    #: The filepath for the file corresponding to the spectrum
    in_filepath: 'pathlib.Path'

    #: The (optional) filepath to write the processed spectrum
    out_filepath: t.Optional['pathlib.Path']

    #: The default attributes that are set to None when reset
    reset_attrs = ('data', 'in_filepath', 'out_filepath')

    def __init__(self, in_filepath, out_filepath=None):
        self.reset()
        self.in_filepath = Path(in_filepath)
        self.out_filepath = (Path(out_filepath)
                             if out_filepath is not None else None)

        # Load the spectrum
        self.load()

    # @property
    # @abc.abstractmethod
    # def ndim(self):
    #     """The number of dimensions in the spectrum"""
    #     raise NotImplementedError

    @abc.abstractmethod
    def load(self, in_filepath: t.Optional['pathlib.Path'] = None):
        """Load the spectrum

        Parameters
        ----------
        in_filepath
            The (optional) filepath to use for loading the spectrum, instead
            of self.in_filepath.
        """
        # Setup arguments
        in_filepath = (Path(in_filepath) if in_filepath is not None else
                       self.in_filepath)

        # Reset attrs, excluding in_filepath and out_filepath
        reset_attrs = tuple(attr for attr in self.reset_attrs
                            if attr not in ('in_filepath', 'out_filepath'))
        self.reset(attrs=reset_attrs)

    def reset(self, attrs: t.Optional[t.Tuple[str, ...]] = None):
        """Reset the data and parameters for the spectrum.

        Parameters
        ----------
        attrs
            A listing of attributes to clear.
        """
        if hasattr(self, 'meta') and isinstance(self.meta, dict):
            self.meta.clear()
        else:
            self.meta = dict()

        # Rest the attributes
        attrs = attrs if attrs is not None else self.reset_attrs
        for attr in attrs:
            setattr(self, attr, None)


class NMRPipeSpectrum(NMRSpectrum):
    """An NMRpipe spectrum"""

    def load(self,
             in_filepath: t.Optional['pathlib.Path'] = None,
             force_nD: bool = False,
             in_plane: str = 'x', out_plane: str = 'DEFAULT'):
        """Load the NMRPipeSpectrum.

        Parameters
        ----------
        force_nD
            If True, force the loading of the nmrPipe spectrum using an
            iterator, which is used for 3Ds, 4Ds, etc.
        in_plane
            The plane read as the direct dimension. e.g. 'x', 'y', 'z', 'a'
        out_plane
             The plane written as the direct dimension. e.g. 'x', 'y', 'z',
             'a', 'DEFAULT'. (Default means it's the same as the in_plane)
        """
        super().load(in_filepath=in_filepath)

        # Determine if the spectrum should be loaded as a series of planes
        # (3D, 4D, etc.) or as and 1D or 2D (plane)
        is_nD = re.search(r'%\d+d', str(self.in_filepath)) is not None

        # Load the spectrum
        if is_nD or force_nD:
            self.data = ng.pipe.iter3D(str(self.in_filepath),
                                       in_plane, out_plane)
            # TODO: Implement assignment of self.meta
            self.is_data_iterator = True
        else:
            dic, data = ng.pipe.read(str(self.in_filepath))
            self.meta = dic
            self.data = data
            self.is_data_iterator = False
