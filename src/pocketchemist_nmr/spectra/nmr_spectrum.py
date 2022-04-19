"""
NMR Spectra in different formats
"""
import abc
import typing as t
from pathlib import Path
from math import floor, ceil, log2

import torch
from torch.nn.functional import pad as pad_tensor
from loguru import logger

from .constants import (UnitType, DomainType, DataType, DataLayout,
                        ApodizationType, RangeType)
from .meta import NMRMetaDict
from .utils import (split_block_to_complex, combine_block_from_complex,
                    split_single_to_complex, combine_single_from_complex,
                    gen_range)

__all__ = ('NMRSpectrum',)


# Abstract base class implementation
class NMRSpectrum(abc.ABC):
    """An NMR spectrum base class.

    Notes
    -----
    - The base class handles the generic processing methodology. Subclasses
      should override methods that are specific to their
      implementation--specifically when interating with the self.meta dict,
      which is implementation specific.
    - The processing functions and some accessors are designed to operate on
      the last (current) dimension only. The other dimensions are processed
      after a transpose to set another dimension as the last dimension. The
      reason for this approach is that NMR multidimensional data is hypercomplex
      such that the last dimension can be viewed as complex numbers but other
      dimension's complex data must interleaved, and it is not complex.
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

    #: The range type for generating ranges of spectrum frequencies
    freq_range_type = RangeType.FREQ | RangeType.ENDPOINT

    #: The range type for generating ranges of FID times.
    time_range_type = RangeType.TIME

    #: The range type for generating ranges for first-order phase correction
    #: sinebell apodization and other methods
    unit_range_type = RangeType.UNIT

    def __init__(self, in_filepath, out_filepath=None):
        self.reset()
        self.in_filepath = Path(in_filepath)
        self.out_filepath = (Path(out_filepath)
                             if out_filepath is not None else None)

        # Load the spectrum
        self.load()

    # Basic accessor/mutator properties

    @property
    def ndims(self) -> int:
        """The number of dimensions in the spectrum"""
        return len(self.data.shape)

    @property
    def npts(self) -> t.Tuple[int, ...]:
        """The number of complex, real + imaginary points in each dimension

        The current dimension is the last dimension.
        """
        return tuple(self.data.size())

    @property
    def npts_data(self) -> t.Tuple[int, ...]:
        """The number of data points, complex or real, in each dimension"""
        # Get the number of complex or real points
        ndims = self.ndims
        return tuple(int(npts / 2)
                     if data_type is DataType.COMPLEX and dim != ndims else
                     npts for dim, (data_type, npts) in
                     enumerate(zip(self.data_type, self.npts), 1))

    @property
    @abc.abstractmethod
    def domain_type(self) -> t.Tuple[DomainType, ...]:
        """The data domain type (freq, time) for all dimensions

        The current dimension is the last dimension."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_type(self) -> t.Tuple[DataType, ...]:
        """The type data (real, imag, complex) of all dimensions

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sw_hz(self) -> t.Tuple[float, ...]:
        """Spectral widths in Hz of all dimensions

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sw_ppm(self) -> t.Tuple[float, ...]:
        """Spectra widths in ppm of all dimensions

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def car_hz(self) -> t.Tuple[float, ...]:
        """The carrier frequencies (in Hz) of all dimensions--i.e. the
        frequency at which the dimension's channel was pulse at.

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def car_ppm(self) -> t.Tuple[float, ...]:
        """The carrier frequencies (in ppm) of all dimensions--i.e. the
        frequency at which the dimension's channel was pulse at.

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def obs_mhz(self) -> t.Tuple[float, ...]:
        """The observed (Zeeman) frequency in MHz of all dimensions

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def range_hz(self) -> t.Tuple[t.Tuple[float, float], ...]:
        """The left- and right-side frequency ranges (in hz) of all dimensions.

        The current dimension is the last dimension

        .. note:: For each dimension, the first range value is greater (>)
                  than the second because NMR data is listed from positive
                  frequencies to negative frequencies.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def range_ppm(self) -> t.Tuple[t.Tuple[float, float], ...]:
        """The left- and right-right frequency ranges (in ppm) of all
        dimensions.

        The current dimension is the last dimension

        .. note:: For each dimension, the first range value is greater (>)
                  than the second because NMR data is listed from positive
                  frequencies to negative frequencies.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def range_s(self) -> t.Tuple[t.Tuple[float, float], ...]:
        """The left- and right-right time ranges (in seconds) of all
        dimensions.

        The current dimension is the last dimension
        """
        raise NotImplementedError


    @property
    def array_hz(self) -> t.Tuple[torch.Tensor, ...]:
        """Generate an array (tensor) of frequency values in Hz for each
        dimension

        The current dimension is the last dimension.
        """
        return tuple(torch.linspace(start=start, end=end, steps=npts)
                     for (start, end), npts
                     in zip(self.range_hz, self.npts_data))

    @property
    def array_ppm(self) -> t.Tuple[torch.Tensor, ...]:
        """Generate an array (tensor) of frequency values in ppm for each
        dimension

        The current dimension is the last dimension.
        """
        return tuple(torch.linspace(start=start, end=end, steps=npts)
                     for (start, end), npts
                     in zip(self.range_ppm, self.npts_data))

    @property
    def array_s(self) -> t.Tuple[torch.Tensor, ...]:
        """Generate an array (tensor) of time values in sec for each dimension

        The current dimension is the last dimension.
        """
        return tuple(torch.linspace(start=start, end=end, steps=npts)
                     for (start, end), npts
                     in zip(self.range_s, self.npts_data))

    @property
    def array_unit(self) -> t.Tuple[torch.Tensor, ...]:
        """Generate an array (tensor) of unit values ([0,1[ or [0,1]) for each
        dimension

        The current dimension is the last dimension.
        """
        return tuple(gen_range(npts, range_type=self.unit_range_type,
                               sw=sw, group_delay=self.group_delay)
                     for npts, sw in zip(self.npts_data, self.sw_hz))

    @property
    @abc.abstractmethod
    def label(self) -> t.Tuple[str, ...]:
        """The labels for all dimensions, as ordered in the data.

        The current dimension is the last dimension.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def apodization(self) -> t.Tuple[ApodizationType, ...]:
        """The type of apodization function applied to each dimension

        The current dimension is the last dimension.
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

    # Accessor functions

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

    def convert(self, value: float,
                unit_from: UnitType = UnitType.POINTS,
                unit_to: UnitType = UnitType.POINTS) -> t.Union[float, int]:
        """Convert a values from one unit to another in the last (current)
        dimension.

        Parameters
        ----------
        value
            The value to convert.
            If the unit_from is in points, negative values indicate the number
            of points from the end of the dataset
        unit_from
            The unit type of the value. eg. UnitType.Hz, UnitType.ppm.
            For UnitType.POINTS, reverse indexing is supported.
        unit_to
            The unit type to convert the value to.

        Returns
        -------
        point
            The point number corresponding to the value
        """
        # Get parameters that will be needed in the calculations
        endpoints = dict()
        npts = self.npts_data[-1]

        for label, unit in (('from', unit_from), ('to', unit_to)):
            if unit is UnitType.POINTS:
                endpoints[label] = (0.0, float(npts - 1))
            elif unit is UnitType.PERCENT:
                endpoints[label] = (0.0, 100.0)
            elif unit is UnitType.SEC:
                endpoints[label] = self.range_s[-1]
            elif unit is UnitType.HZ:
                endpoints[label] = self.range_hz[-1]
            elif unit is UnitType.PPM:
                endpoints[label] = self.range_ppm[-1]
            else:
                raise NotImplementedError

        # Get the point position for the 'from' value.
        if unit_from is UnitType.POINTS and value < 0:
            # Allow reverse indexing for unit_from UnitType.POINTS
            point = float(npts) + float(value)
        else:
            # Otherwise, calculate the new point value from the endpoints
            point = ((float(value) - endpoints['from'][0]) * float(npts - 1) /
                     (endpoints['from'][1] - endpoints['from'][0]))

        # Check that the number of points is within range
        assert 0.0 <= point < float(npts), (
            f"The value '{value}' is not within range of "
            f"[{endpoints['from'][0]}, {endpoints['from'][1]}[")

        # Convert the point position to the new value
        new_value = ((endpoints['to'][1] - endpoints['to'][0]) *
                     (round(point) / (npts - 1)) + endpoints['to'][0])

        return round(new_value) if unit_to is UnitType.POINTS else new_value

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
                        update_meta: bool = True) -> None:
        """Apply exponential apodization to the last dimension.

        The apodization function scales the time-domain signal, or free
        induction decay (:math:`f(t)`), by an exponential decay:

        .. math::
            g(t) = e^{-\\pi \\mathtt{lb} \\cdot x} f(t)

        The Fourier transform is convolved with a Lorentzian function's shape.
        By default, the 'x' axis range is in time units, [0, tmax[, but may
        optionally be in other unit types.

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
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.

        Notes
        -----
        - NMRPipe implements start/size before zeroing points outside this
          range. This function, instead, copies these points (scale 1.0) and
          apodizes points within the range.
        """
        # Prepare arguments
        t = self.array_s[-1]  # Get last (current) dim
        size = int(self.npts[-1]) if size is None else size

        # Calculate the decay rate
        k = torch.ones(len(t))
        k[start:start + size] = torch.abs(lb * torch.pi *
                                          t[start: start + size])

        # Calculate the apodization func
        self.data *= torch.exp(-k)

    def apodization_sine(self,
                         off: float = 0.5,
                         end: float = 1.0,
                         power: float = 1.0,
                         first_point_scale: float = 1.0,
                         start: int = 0, size: t.Optional[int] = None,
                         update_meta: bool = True) -> None:
        """Apply sine-bell apodization to the last dimension.

        The apodization function scales the time-domain signal, or free
        induction decay (:math:`f(t)`), by a sine function:

        .. math::
            g(t) = \\sin(-\\pi (\\mathtt{off} + (\\mathtt{end} - \\mathtt{off})
            \\cdot x))^{\\mathtt{power}} f(t)

        The Fourier transform is convolved with a cosine function's shape.
        By default, the range for the 'x' axis is a unit range, [0,1[, but
        other types are available.

        Parameters
        ----------
        off
            Offset to start of the sine-bell function as a factor of pi.
        end
            End of the sine-bell function as a factor of pi.
        power
            The exponent of the sine-belle.
        first_point_scale
            Scale the first point by this number
        start
            Apply apodization starting from this point
        size
            Apply apodization over this length of points
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.

        Notes
        -----
        - NMRPipe implements start/size before zeroing points outside this
          range. This function, instead, copies these points (scale 1.0) and
          apodizes points within the range.
        """
        # Prepare arguments
        x = self.array_unit[-1]  # Get last (current) dim
        size = int(self.npts[-1]) if size is None else size

        # Calculate the sine-belle function
        off *= torch.pi
        end *= torch.pi
        k = torch.ones(len(x))
        k[start:start + size] = torch.sin(off +
                                          (end - off) *
                                          x[start: start + size]) ** power

        # Calculate the apodization func
        self.data *= k

    def extract(self,
                start: t.Union[int, float],
                unit_start: UnitType,
                end: t.Union[int, float],
                unit_end: UnitType,
                update_meta: bool = True):
        """Extract a region of the last (current) dimension.

        Parameters
        ----------
        start
            Extracted region starting position
        unit_start
            The unit of the starting position value
        end
            Extracted region ending position
        unit_end
            The unit of the ending position value
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.

        Returns
        -------
        start_point, end_point
            The starting and end point of the extraction
        """
        # Check that the unit types are compatible with the domain type
        if self.domain_type[-1] is DomainType.FREQ:
            if UnitType.SEC in (unit_start, unit_end):
                raise ValueError(f"The start and end units must be in points "
                                 f"or frequency units for a frequency domain")
        elif self.domain_type[-1] is DomainType.TIME:
            if (UnitType.HZ in (unit_start, unit_end) or
               UnitType.PPM in (unit_start, unit_end)):
                raise ValueError(f"The start and end units must be in points "
                                 f"or time units for a time domain")

        # Get the starting and ending points
        start_point = self.convert(value=start, unit_from=unit_start,
                                   unit_to=UnitType.POINTS)
        end_point = self.convert(value=end, unit_from=unit_end,
                                 unit_to=UnitType.POINTS)
        start_point, end_point = (min(start_point, end_point),
                                  max(start_point, end_point))  # reorder

        if start_point > 0:
            start_point -= 1  # Req to get data of size [start_point, end_point]

        # Conduct the extraction
        self.data = self.data[..., start_point: end_point]

        return start_point, end_point

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
        """Perform a Fourier Transform to the last (current) dimension.

        The discrete fast Fourier transformation (fft) in the forward
        (time -> frequency) direction is as follows:

        .. math::
            X_k = \\sum_0^{N-1} x_n \\cdot e^{- i 2 \\pi k n / N}

        And the inverse discrete Fourier transformation (ifft) in the backward
        (frequency -> time) direction is as follows:

        .. math::
            x_n = \\frac{1}{N} \\sum_0^{N-1} X_k \\cdot e^{i 2 \\pi k n / N}

        The time series 'n' or frequency series 'k' start at 0 and increase to
        the largest positive value 'N'. NMR data typically places
        the 0 Hz frequency component in the middle with positive frequencies
        on the left and negative frequencies on the right.

        .. math::
            k = \\Big[ \\frac{sw}{2}, ..., 0Hz, ..., -\\frac{sw}{2} \\Big]

        The centering of frequencies is achieved with the 'center' option.

        Flipping the sign of the frequencies can be achieved by using ifft
        instead of fft. However, to more closely match the FT behavior of
        processing software with digital filter correction, like NMRPipe, the
        default uses fft as usual for time-domain data, and the FT data are
        'flipped' with the 'flip' option.

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
        if self.correct_digital_filter:
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
        if self.correct_digital_filter:
            group_delay = self.group_delay
            # The p0/p1 selected here more closely matches NMRPipe
            # The p0=180. likely arises from the flip below.
            p1 = 1. * (group_delay - floor(group_delay)) * 360.  # degrees
            self.phase(p0=180.0, p1=p1, discard_imaginaries=False,
                       update_meta=False)

        # Flip the last dimension, if needed
        if flip:
            self.data = torch.flip(self.data, (-1,))

    def phase(self, p0: float, p1: float,
              discard_imaginaries: bool = True,
              update_meta: bool = True):
        """Apply phase correction to the last dimension.

        Phasing rotates the real and imaginary components of complex numbers
        by a specified phase. This function may apply frequency-independent
        phase angle corrections (p0) as well as linear freqency-dependent
        phase angle corrections (p1).

        .. math::
            g(t) = e^{i(\\mathtt{p0} + \\mathtt{p1} \\cdot x)} g(t)
        .. math::
            G(\\omega) = e^{i(\\mathtt{p0} + \\mathtt{p1} \\cdot x)} f(\\omega)

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
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.
        """
        # Prepare arguments
        x = self.array_unit[-1]  # Get last (current) dim

        # Apply the zeroth and first order phase
        phase = p0 + p1 * x
        phase *= torch.pi / 180.  # in radians
        self.data *= torch.exp(phase * 1.j)

        if discard_imaginaries:
            self.data = self.data.real

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

    def zerofill(self,
                 double: t.Optional[int] = 1,
                 double_base2: t.Optional[int] = None,
                 size: t.Optional[int] = None,
                 pad: t.Optional[int] = None,
                 update_meta: bool = True) -> None:
        """Zero-fill the last dimension

        Parameters
        ----------
        double
            The number of times to double the size of the last dimension with
            zero-filling
        double_base2
            The number of times to double the size of the last dimension to
            match the next 2^N size
        size
            The final size after zero-filling
        pad
            The number of points to add with zero-filling
        update_meta
            Update the meta dict. This functionality is handled by sub-classes.
        """
        # Determine the number of points to zero-fill to
        npts = int(self.data.size()[-1])
        if size is not None:
            pass
        elif pad is not None:
            size = npts + pad
        elif double_base2 is not None:
            size = 2**(ceil(log2(npts)) + float(double_base2) - 1.0)
        else:
            size = npts * 2 * double

        # Zero-fill (pad) the last dimension with zeroes
        delta = int(size - npts)  # Number of points to pad
        assert delta > 0, (f"Zero-filling only works if the final number of "
                           f"points {size} is larger than the current number "
                           f"of points {size}")
        self.data = pad_tensor(self.data, pad=(0, delta),  # pad_left, pad_right
                               mode='constant', value=0.0)
