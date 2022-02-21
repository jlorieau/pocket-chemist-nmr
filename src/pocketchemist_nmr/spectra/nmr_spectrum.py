"""
NMR Spectra in different formats
"""
import abc
import typing as t
from pathlib import Path
from math import floor

import torch
from loguru import logger

from .constants import (DomainType, DataType, DataLayout, ApodizationType,
                        RangeType)
from .meta import NMRMetaDict
from .utils import (split_block_to_complex, combine_block_from_complex,
                    split_single_to_complex, combine_single_from_complex,
                    gen_range)

__all__ = ('NMRSpectrum',)


# Abstract base class implementation
class NMRSpectrum(abc.ABC):
    """An NMR spectrum base class.

    The base class handles the generic processing methodology. Subclasses
    should override methods that are specific to their
    implementation--specifically when interating with the self.meta dict, which
    is implementation specific.
    """

    #: Metadata on the spectrum.
    #: All methods should maintain the correct integrity of the metadata.
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

    @property
    @abc.abstractmethod
    def apodization(self) -> t.Tuple[ApodizationType, ...]:
        """The type of apodization function applied to each dimension"""
        raise NotImplementedError

    @property
    def npts(self) -> t.Tuple[int, ...]:
        """The number of complex, real or imaginary points in each dimension"""
        return tuple(self.data.size())

    @abc.abstractmethod
    def data_layout(self, dim: int,
                    data_type: t.Optional[DataType] = None) -> DataLayout:
        """Give the current data layout for all dimensions or the expected
        data layout for the given data type and dimension.

        Parameters
        ----------
        dim
            The current or expected data layout for the given dimension
        data_type
            If specified, give the expected data layout for the given data type

        Returns
        -------
        data_layout
            The data layout for the given data type and dimension
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def group_delay(self) -> t.Union[None, float]:
        """The digital filter group delay for the last dimension, if it was
        passed through a moving average digital filter.

        Returns
        -------
        group_delay
            The value of the group delay in number of points, if a digital
            filter was applied in the dimension, or none if there is no
            digital filter in this dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def correct_digital_filter(self) -> bool:
        """Whether a digital correction filter must be corrected (removed)
        from the last dimension."""
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
    def apodization_exp(self, lb: float, first_point_scale: float = 1.0,
                        start: int = 0, size: t.Optional[int] = None,
                        range_type: RangeType = RangeType.TIME,
                        update_meta: bool = True) -> None:
        """Apply exponential apodization to the last dimension.

        The apodization function scales the time-domain signal, or free
        induction decay (:math:`f(t)`), by an exponential decay:

        .. math::
            g(t) = e^{-\pi lb \cdot t} f(t)

        The Fourier transform is convolved with a Lorentzian function's shape.

        Parameters
        ----------
        lb
            Exponential decay constant (in Hz)
        first_point_scale
            Scale the first point by this number
        start
            Apply apodization starting from this point
        size
            Apply apodization over this length of points
        range_type
            The type of range to use for the x-axis data points
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.
        """
        # Get the time delays
        sw = self.sw[-1]  # Spectral width (Hz)
        npts = self.npts[-1]  # Number of points
        size = npts if size is None else size
        scaled_sw = sw * float(size - start) / float(npts)
        size = npts if size is None else npts
        group_delay = self.group_delay if self.correct_digital_filter else 0.0

        # Calculate the decay rate
        k = torch.ones(npts)
        t = gen_range(npts=size, range_type=range_type, sw=scaled_sw,
                      group_delay=group_delay)
        k[start:start + size] = torch.abs(lb * torch.pi * t)

        # Calculate the apodization func
        self.data *= torch.exp(-k)

    def transpose(self, dim0: int, dim1: int, update_data_layout: bool = True,
                  update_meta: bool = True):
        """Transpose two axes (dim0 <-> dim1).

        Parameters
        ----------
        dim0
            The first dimension to transpose, starting from 0 to self.ndims - 1
        dim1
            The second dimension to transpose, starting from 0 to self.ndims - 1
        update_data_layout
            If True (default), automatically convert conplex numbers and handle
            changes in data layout according to the
            :meth:`.NMRSpectrum.data_layout` method.
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.
        """
        # Only works if there is more than 1 dimension
        assert self.ndims > 1, (
            "Can only transpose spectrum with more than 1 dimension")

        # Sort the order of the dimensions
        dim0, dim1 = min(dim0, dim1), max(dim0, dim1)

        # # Unpack complex numbers in the last dimension, if needed
        if (update_data_layout and dim1 == self.ndims - 1 and
           self.data_type[dim1] is DataType.COMPLEX):
            # Determine the data layout the new dimension will need
            new_data_layout = self.data_layout(dim0, data_type=DataType.COMPLEX)
            logger.debug(f"new_data_layout: {new_data_layout}")

            if new_data_layout is DataLayout.BLOCK_INTERLEAVE:
                self.data = combine_block_from_complex(self.data)
            elif new_data_layout is DataLayout.SINGLE_INTERLEAVE:
                self.data = combine_single_from_complex(self.data)
            else:
                raise NotImplementedError

        # Conduct the transpose
        self.data = torch.transpose(self.data, dim0, dim1)

        # Determine if the new last dimension should be converted to complex
        if (update_data_layout and dim1 == self.ndims - 1 and
           self.data_type[dim0] is DataType.COMPLEX):
            # Determine the data layout for the old dimension
            old_data_layout = self.data_layout(dim0, data_type=DataType.COMPLEX)
            logger.debug(f"old_data_layout: {old_data_layout}")

            if old_data_layout is DataLayout.BLOCK_INTERLEAVE:
                self.data = split_block_to_complex(self.data)
            elif old_data_layout is DataLayout.SINGLE_INTERLEAVE:
                self.data = split_single_to_complex(self.data)
            else:
                raise NotImplementedError

    def phase(self, p0: float, p1: float, discard_imaginaries: bool = True,
              range_type: RangeType = RangeType.UNIT, update_meta: bool = True):
        """Apply phase correction to the last dimension.

        Phasing rotates the real and imaginary components of complex numbers
        by a specified phase. This function may apply frequency-independent
        phase angle corrections (p0) as well as linear freqency-dependent
        phase angle corrections (p1).

        .. math::
            g(t) = e^{i(p0 + p1 \cdot x)} g(t)
        .. math::
            G(\omega) = e^{i(p0 + p1 \cdot x)} f(\omega)

        Where the x-axis range type may be changed from unit_type ([0, 1[),
        time or some other unit.

        Parameters
        ----------
        p0
            The zero-order phase correction (in degrees)
        p1
            The first-order phase correction (in degrees / Hz)
        discard_imaginaries
            Only keep the real component of complex numbers after phase
            correction and discard the imaginary component
        range_type
            The type of range to use for the x-axis data points
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.
        """
        # Get the spectra width and data length for the last dimension
        sw = self.sw[-1]
        npts = self.data.size()[-1]
        group_delay = self.group_delay if self.correct_digital_filter else 0.0

        x = gen_range(npts, range_type=range_type, sw=sw,
                      group_delay=group_delay)
        phase = p0 + p1 * x
        phase *= torch.pi / 180.  # in radians
        self.data *= torch.exp(phase * 1.j)

        if discard_imaginaries:
            self.data = self.data.real

    def ft(self,
           auto: bool = False,
           center: bool = True,
           flip: bool = True,
           real: bool = False,
           inv: bool = False,
           alt: bool = False,
           neg: bool = False,
           bruk: bool = False,
           update_meta: bool = True):
        """Perform a Fourier Transform to the last dimension.



        Parameters
        ----------
        auto
            Try to determine the FT flags automatically
        center
            If True (default), shift the 0Hz frequency component to the center
        flip
            If True (default), flip the ordering of data in reverse order.
            This function is needed for NMR FT to place positive frequencies
            before negative frequencies.
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
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.

        Returns
        -------
        kwargs
            The kwargs dict with the 'data' entry populated with the Fourier
            Transformed dataset.

        Notes
        -----
        The digital correction of this function doesn't give the same results
        as NMRPipe: The final zeroth order phase is different by 10s of degrees,
        but the first-order phase appears to match.
        """
        # Setup the arguments
        fft_func = torch.fft.fft
        fft_shift = torch.fft.fftshift

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
            fft_func = torch.fft.ifft
            fft_shift = torch.fft.ifftshift
        if alt and not inv:
            # Alternate the sign of points
            self.data[..., 1::2] *= -1.
        if neg:
            # Negate (multiple by -1) the imaginary component
            self.data.imag *= -1.0

        logger.debug(f"auto: {auto}, center: {center}, flip: {flip}, "
                     f"real: {real}, inv: {inv}, alt: {alt}, neg: {neg}, "
                     f"bruk: {bruk}, "
                     f"correct_digital_filter: {self.correct_digital_filter}")

        # Remove digitization, if needed
        if self.correct_digital_filter and not inv:
            shift_points = int(floor(self.group_delay))
            self.data = torch.roll(self.data, (-shift_points))

        # Perform the FFT then a frequency shift
        if center:
            # Apply fft_shift on the last dimension, which is the one being
            # Fourier transformed
            self.data = fft_shift(fft_func(self.data), dim=-1)
        else:
            self.data = fft_func(self.data)

        # Post process the data
        if inv and alt:
            self.data[..., 1::2] = self.data[..., 1::2] * -1

        # Apply digitization phase shift, if needed
        if self.correct_digital_filter and not inv:
            group_delay = self.group_delay
            # The p0/p1 selected here more closely matches NMRPipe
            # The p0=180. likely arises from the flip below.
            p1 = 1. * (group_delay - floor(group_delay)) * 360.  # degrees
            self.phase(p0=180.0, p1=p1, discard_imaginaries=False,
                       update_meta=False)

        # Flip the last dimension, if needed
        if flip:
            self.data = torch.flip(self.data, (-1,))
