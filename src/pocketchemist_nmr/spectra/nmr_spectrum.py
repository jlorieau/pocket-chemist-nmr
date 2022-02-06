"""
NMR Spectra in different formats
"""
import abc
import typing as t
from pathlib import Path

from torch import permute, fft

from .constants import DomainType, DataType
from .meta import NMRMetaDict
from .utils import split_to_complex, combine_from_complex

__all__ = ('NMRSpectrum',)


# Abstract base class implementation
class NMRSpectrum(abc.ABC):
    """An NMR spectrum base class.

    .. note::
          The base class handles the generic processing methodology.
          Subclasses should override methods that are specific to their
          implementation--specifically when interating with the self.meta
          dict, which is implementation specific.
    """

    #: metadata on the spectrum.
    #: All methods should maintain the correct integrity of the metadata
    meta: NMRMetaDict

    #: The data for the spectrum, either an array or an iterator
    data: 'torch.Tensor'

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

    # Basic accessor/mutator methods

    @property
    @abc.abstractmethod
    def ndims(self) -> int:
        """The number of dimensions in the spectrum"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def domain_type(self) -> t.Tuple[DomainType, ...]:
        """The data domain type (freq, time) for all available dimensions, as
        ordered in the data.

        Returns
        -------
        domain_type
            The current value of the domain type setting.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_type(self) -> t.Tuple[DataType, ...]:
        """The type data (real, imag, complex) of all available dimensions, as
        ordered in the data."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sw(self) -> t.Tuple[int, ...]:
        """Spectral widths (in Hz) of all available dimensions, as ordered in
        the data."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def label(self) -> t.Tuple[str, ...]:
        """The labels for all dimensions, as ordered in the data."""
        raise NotImplementedError

    # I/O methods

    @abc.abstractmethod
    def load(self, in_filepath: t.Optional['pathlib.Path'] = None):
        """Load the spectrum

        Parameters
        ----------
        in_filepath
            The (optional) filepath to use for loading the spectrum, instead
            of self.in_filepath.
        """
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
        if hasattr(self, 'meta') and hasattr(self.meta, 'clear'):
            self.meta.clear()
        else:
            # Create a new meta dict based on the annotation
            meta_cls = t.get_type_hints(self)['meta']
            self.meta = meta_cls()

        # Rest the attributes
        attrs = attrs if attrs is not None else self.reset_attrs
        for attr in attrs:
            setattr(self, attr, None)

    # Manipulator methods

    def permute(self, new_dims: t.Tuple[int, ...],
                interleave=True):
        """Permute (transpose) axes according to the new dimension order.

        Parameters
        ----------
        new_dims
            The new order of the dimensions. Dimensions are ordered from
            0 to (self.ndims - 1).
        interleave
            Split/stack the inner dimension between complex and real points,
            depending on whether the dimension is complex or not.
        """
        # Only works if there is more than 1 dimension
        assert self.ndims > 1, (
            "Can only permute multiple dimensions")

        # Get the datatypes for each dimension before and after permute
        before_data_types = self.data_type
        after_data_types = tuple(data_type for _, data_type in
                                 sorted(zip(new_dims, before_data_types)))

        # If the last dimension is complex, stack the real/imag points
        # into a set of real points before permuting
        if interleave and before_data_types[-1] == DataType.COMPLEX:
            self.data = combine_from_complex(self.data)

        # Reorganize data
        self.data = permute(self.data, new_dims)

        # If the last dimension should be complex, split the stack of real/imag
        # points into a complex dimension
        if interleave and after_data_types[-1] == DataType.COMPLEX:
            self.data = split_to_complex(self.data)

    def ft(self,
           auto: bool = False,
           real: bool = False,
           inv: bool = False,
           alt: bool = False,
           neg: bool = False,
           bruk: bool = False,
           **kwargs):
        """Perform a Fourier Transform

        This method is designed to be used on instances and as a class method.

        Parameters
        ----------
        ft_func
            The Fourier Transform wrapper functions to use.
        auto
            Try to determine the FT flags automatically
        real
            Apply a real Fourier transform (.FFTType.RFFT)
        inv
            Apply an inverse Fourier transform (.FFTType.IFFT)
        alt
            Alternate the sign of points before Fourier transform
        neg
            Negate imaginary component of complex numbers before Fourier
            transform
        bruk
            Process Redfield sequential data, which is alt and real.
        meta
            Metadata on the spectrum
        data
            The data to Fourier Transform

        Returns
        -------
        kwargs
            The kwargs dict with the 'data' entry populated with the Fourier
            Transformed dataset.

        See Also
        --------
        - nmrglue.process.proc_base
        """
        # Setup the arguments
        fft_func = fft.fft

        # Setup the flags
        if auto:
            # The auto flag should be set to False when this method is called
            # by children methods. Children methods are responsible for
            # determining how to apply and 'auto' processing
            raise NotImplementedError

        if bruk:
            # Adjust flags for Redfield sequential data
            real = True
            alt = True
        if real:
            # Remove the imaginary component for real transformation
            self.data.imag = 0.0
        if inv:
            # Set the FFT function type to inverse Fourier transformation
            fft_func = fft.ifft
        if alt and not inv:
            # Alternate the sign of points
            self.data[..., 1::2] = self.data[..., 1::2] * -1.
        if neg:
            # Negate (multiple by -1) the imaginary component
            self.data.imag *= -1.0

        # Perform the FFT then a frequency shift
        self.data = fft_func(data=self.data)

        # Post process the data
        if inv and alt:
            self.data[..., 1::2] = self.data[..., 1::2] * -1

        # Prepare the return value
        return kwargs
