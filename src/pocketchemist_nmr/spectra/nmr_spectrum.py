"""
NMR Spectra in different formats
"""
import abc
import typing as t
from pathlib import Path

__all__ = ('NMRSpectrum',)


class NMRSpectrum(abc.ABC):
    """An NMR spectrum base class."""

    #: metadata on the spectrum
    meta: dict

    #: The data for the spectrum, either an array or an iterator
    data: t.Union[t.Iterable, 'numpy.ndarray']

    #: If True, the data represents multiple files that must be iterated over
    is_multifile: bool = False

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

    @property
    @abc.abstractmethod
    def ndims(self) -> int:
        """The number of dimensions in the spectrum"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def order(self) -> t.Tuple[int, ...]:
        """The order of the dimensions for the data.

        Returns
        -------
        order
            The ordering of the dimensions for the :attr:`data` dataset.
            - The first dimension corresponds to the rows of the dataset
              (axis=1 for numpy)
            - The second dimension corresponds to the columns of the dataset
              (axis=0 for numpy)
            - The third and higher dimensions corresponds to additional
              dimensions, which may be represented by an iterator
        """
        raise NotImplementedError

    @order.setter
    @abc.abstractmethod
    def order(self, new_order: t.Tuple[int, ...]):
        """Change the order of the dimensions for the data.

        Parameters
        ----------
        new_order
            The new order of the dataset. This function is typically invoked
            for transpose operations.
        """
        raise NotImplementedError

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

    @abc.abstractmethod
    def save(self,
             out_filepath: t.Optional['pathlib.Path'] = None,
             format: str = None,
             overwrite: bool = True):
        """Save the spectrum to the specified filepath

        Parameters
        ----------
        out_filepath
            The filepath for the file(s) to save the spectrum.
        format
            The format of the spectrum to write. By default, this is nmrpipe.
        overwrite
            If True (default), overwrite existing files.
        """
        pass

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
