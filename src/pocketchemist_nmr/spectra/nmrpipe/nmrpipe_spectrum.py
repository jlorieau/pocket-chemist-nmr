"""
NMRSpectrum in NMRPipe format
"""
import re
import typing as t
from pathlib import Path
from functools import reduce

from loguru import logger

from .constants import Plane2DPhase, SignAdjustment, find_mapping
from .fileio import (load_nmrpipe_tensor, load_nmrpipe_multifile_tensor,
                     save_nmrpipe_tensor)
from ...filters.bruker import bruker_group_delay
from .meta import NMRPipeMetaDict
from ..nmr_spectrum import NMRSpectrum
from ..utils import range_endpoints
from ..constants import (UnitType, DomainType, DataType, DataLayout,
                         ApodizationType, RangeType)

__all__ = ('NMRPipeSpectrum',)


# Concrete subclass
class NMRPipeSpectrum(NMRSpectrum):
    """An NMRpipe spectrum

    Attributes
    ----------
    meta
        The dict containing header values for the NMRPipe spectrum. It
        includes:

        From: http://rmni.iqfr.csic.es/HTML-manuals/nmrpipe-manual/fdatap.html

        - 'FDDIMCOUNT': Number of dimensions in the complete spectrum
        - 'FDDIMORDER': The ordering of dimensions for the data attribute.
          The ordering starts at 1 and increases for each other dimension.
          Dimensions 1, 2, 3 and 4 represent the x-, y-, z- and a-axes,
          respectively.
        - 'FDFxSW': The spectral width (in Hz) for dimension 'x'
        - 'FDFxFTFLAG': Whether the dimension 'x' is in the frequency domain
          (1.0) or the time domain (0.0)
        - 'FD2DPHASE': Describes the type of 2D file plane, if the data is 2-,
          3-, 4-dimensional.

    .. note:: NMRPipe orders data in the file as inner-outer1-outer2 wherease
              torch tensors are setup as outer2-outer1-inner. Consequently,
              the order of dimensions in the methods are reversed.
    """
    #: Header metadata dict
    meta: NMRPipeMetaDict

    #: The range type for generating ranges of times. The group delay
    #: correction will be applied for the last dimension when
    #: correct_digital_filter is True
    time_range_type = RangeType.TIME | RangeType.GROUP_DELAY

    # Basic accessor/mutator methods

    @property
    def order(self) -> t.Tuple[int, ...]:
        """The ordering of the data dimensions to the F1/F2/F3/F4 channels
        in the header.

        The order is a value between 1 and 4.
        """
        fddimorder = [int(self.meta[f"FDDIMORDER{dim}"]) for dim in range(1, 5)]

        # Swap order. Tenors are stored outer-inner while NMRPipe is stored
        # inner-outer
        return tuple(fddimorder[:self.ndims][::-1])

    @property
    def domain_type(self) -> t.Tuple[DomainType, ...]:
        # Setup mappings between DomainTypes and the meta dict values
        domain_types = []
        for dim in self.order:
            value = self.meta[f"FDF{dim}FTFLAG"]
            domain_types.append(find_mapping('domain_type', value))
        return tuple(domain_types)

    @property
    def data_type(self) -> t.Tuple[DataType, ...]:
        # Setup mappings between DataTypes and the meta dict values
        data_types = []
        for dim in self.order:
            value = self.meta[f"FDF{dim}QUADFLAG"]
            data_types.append(find_mapping('data_type', value))
        return tuple(data_types)

    @property
    def sw_hz(self) -> t.Tuple[float, ...]:
        return tuple(self.meta[f"FDF{dim}SW"] for dim in self.order)

    @property
    def sw_ppm(self) -> t.Tuple[float, ...]:
        return tuple(sw / obs for sw, obs in zip(self.sw_hz, self.obs_mhz))

    @property
    def car_hz(self) -> t.Tuple[float, ...]:
        return tuple(car * obs for car, obs in zip(self.car_ppm, self.obs_mhz))

    @property
    def car_ppm(self) -> t.Tuple[float, ...]:
        return tuple(self.meta[f"FDF{dim}CAR"] for dim in self.order)

    @property
    def obs_mhz(self) -> t.Tuple[float, ...]:
        return tuple(self.meta[f"FDF{dim}OBS"] for dim in self.order)

    @property
    def range_hz(self) -> t.Tuple[t.Tuple[float, float], ...]:
        # FDF{dim}ORIG is the Hz frequency of the last point.
        range_hz = []
        for dim in self.order:
            orig_hz = self.meta[f"FDF{dim}ORIG"]
            sw_hz = self.meta[f"FDF{dim}SW"]
            npts = (-1. * self.meta[f"FDF{dim}ZF"]  # Use ZF size, if available
                    if self.meta[f"FDF{dim}ZF"] < -1.0 else
                    self.meta[f"FDF{dim}TDSIZE"])  # Otherwise use TDSIZE
            start, end = range_endpoints(npts=npts,
                                         range_type=self.freq_range_type,
                                         sw=sw_hz, group_delay=self.group_delay)

            range_left = orig_hz + end
            range_right = orig_hz

            range_hz.append((range_left, range_right))
        return tuple(range_hz)

    @property
    def range_ppm(self) -> t.Tuple[t.Tuple[float, float], ...]:
        return tuple((rng[0] / obs_mhz, rng[1] / obs_mhz)
                     for rng, obs_mhz in zip(self.range_hz, self.obs_mhz))

    @property
    def range_s(self) -> t.Tuple[t.Tuple[float, float], ...]:
        range_s = []
        for count, dim in enumerate(self.order, 1):
            sw_hz = self.meta[f"FDF{dim}SW"]

            if (self.meta[f"FDF{dim}X1"] > 0.0 and
               self.meta[f"FDF{dim}XN"] > 0.0):
                # Get the data size from the extracted region first
                npts = (int(self.meta[f"FDF{dim}XN"]) -
                        int(self.meta[f"FDF{dim}X1"]) + 1)
            elif self.meta[f"FDF{dim}ZF"] < -1.0:
                # Then get the npts from the zero-fill
                npts = -1. * self.meta[f"FDF{dim}ZF"]
            else:
                # Finally get the npts from the original data size
                npts = self.meta[f"FDF{dim}TDSIZE"]

            # Determine whether the group delay correction must be applied to
            # the last dimension
            if self.correct_digital_filter and count == self.ndims:
                start, end = range_endpoints(npts=npts,
                                             range_type=self.time_range_type,
                                             sw=sw_hz,
                                             group_delay=self.group_delay)
            else:
                start, end = range_endpoints(npts=npts,
                                             range_type=self.time_range_type,
                                             sw=sw_hz)

            range_s.append((start, end))
        return tuple(range_s)

    @property
    def label(self) -> t.Tuple[str, ...]:
        return tuple(self.meta[f"FDF{dim}LABEL"] for dim in self.order)

    @property
    def apodization(self) -> t.Tuple[ApodizationType, ...]:
        # Setup mappings between ApodizationType and the meta dict values
        apodization = []
        for dim in self.order:
            value = self.meta[f"FDF{dim}APODCODE"]
            apodization.append(find_mapping('apodization', value))
        return tuple(apodization)

    @property
    def group_delay(self) -> (t.Union[None, float], bool):
        # Try getting Bruker's group delay
        return bruker_group_delay(grpdly=self.meta.get('FDDMXVAL', None))

    @property
    def correct_digital_filter(self) -> bool:
        # Determine if the Bruker digitization hasn't yet been applied
        dmxflag = round(self.meta.get('FDDMXFLAG', -1.0))
        return (False if dmxflag == -1.0 or  # DMX ON
                dmxflag == 1.0 else  # DMX OFF
                True)  # DMX auto

    def data_layout(self, dim: int,
                    data_type: t.Optional[DataType] = None) -> DataLayout:
        # For NMRPipe, the last dimension (inner loop) is block interleaved
        # when complex whereas other dimensions as single interleaved (outer
        # loops) when complex.
        data_type = self.data_type[dim] if data_type is not None else data_type

        if data_type in (DataType.REAL, DataType.IMAG):
            return DataLayout.CONTIGUOUS
        elif dim == self.ndims - 1 and data_type is DataType.COMPLEX:
            # The last dimension's data layout is different
            return DataLayout.BLOCK_INTERLEAVE
        elif data_type is DataType.COMPLEX:
            return DataLayout.SINGLE_INTERLEAVE
        else:
            raise NotImplementedError

    @property
    def sign_adjustment(self) -> t.Tuple[SignAdjustment, ...]:
        """The type of sign adjustment needed for each dimension.
        """
        sign_adjustments = []
        for dim in self.order:
            value = self.meta[f'FDF{dim}AQSIGN']
            sign_adjustments.append(find_mapping('sign_adjustment', value))
        return tuple(sign_adjustments)

    @property
    def plane2dphase(self):
        """The phase of 2D planes for 2-, 3-, 4-dimensional self.data values."""
        return find_mapping('plane2dphase', self.meta['FD2DPHASE'])

    # I/O methods

    def load(self,
             in_filepath: t.Optional[t.Union[str, Path]] = None,
             shared: bool = True,
             device: t.Optional[str] = None,
             force_gpu: bool = False):
        """Load the NMRPipeSpectrum.

        Parameters
        ----------
        in_filepath
            The filepath for the spectrum file(s) to load.
        shared
            Create the tensor storage to be shared between threads/processing
        device
            The name of the device to allocate the memory on.
        force_gpu
            Force allocating the tensor on the GPU
        """
        super().load(in_filepath=in_filepath)

        # Determine if the spectrum should be loaded as a series of planes
        # (3D, 4D, etc.) or as and 1D or 2D (plane)
        is_multifile = re.search(r'%\d+d', str(self.in_filepath)) is not None

        # Load the spectrum and assign attributes
        if is_multifile:
            # Load the tensor from multiple files
            meta_dicts, data = load_nmrpipe_multifile_tensor(
                filemask=str(self.in_filepath), shared=shared, device=device,
                force_gpu=force_gpu)
            self.meta, self.data = meta_dicts[0], data
        else:
            meta, data = load_nmrpipe_tensor(filename=str(self.in_filepath),
                                             shared=shared, device=device,
                                             force_gpu=force_gpu)
            self.meta, self.data = meta, data

    def save(self,
             out_filepath: t.Optional[t.Union[str, Path]] = None,
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
        # Setup arguments
        out_filepath = (out_filepath if out_filepath is not None else
                        self.out_filepath)
        save_nmrpipe_tensor(filename=out_filepath, meta=self.meta,
                            tensor=self.data, overwrite=overwrite)

    # Manipulator methods

    def update_meta(self):
        """Update the meta dict with the current values of the data"""
        # Update the data intensity ranges
        if self.data.is_complex():
            self.meta['FDMAX'] = float(self.data.real.max())
            self.meta['FDMIN'] = float(self.data.real.min())
        else:
            self.meta['FDMAX'] = float(self.data.max())
            self.meta['FDMIN'] = float(self.data.min())
        self.meta['FDDISPMAX'] = self.meta['FDMAX']
        self.meta['FDDISPMIN'] = self.meta['FDMIN']

        # The FDMIN/FDMAX are valid
        self.meta['FDSCALEFLAG'] = 1.0

        # Update the datatype for the current (last) dimension
        self.meta['FDQUADFLAG'] = find_mapping('data_type', self.data_type[-1],
                                               reverse=True)

        # Update the number of points for the last (inner) dimension
        self.meta['FDSIZE'] = float(self.data.size()[-1])

    def apodization_exp(self, lb: float, first_point_scale: float = 1.0,
                        start: int = 0, size: t.Optional[int] = None,
                        update_meta: bool = True):
        super().apodization_exp(lb=lb, first_point_scale=first_point_scale,
                                start=start, size=size, update_meta=update_meta)

        # Update the metadata values
        if update_meta:
            dim = self.order[-1]
            new_apod_code = find_mapping('apodization',
                                         ApodizationType.EXPONENTIAL,
                                         reverse=True)
            self.meta[f"FDF{dim}APODCODE"] = float(new_apod_code)
            self.meta[f"FDF{dim}APODQ1"] = float(lb)

            # Update other meta dict values
            self.update_meta()

    def apodization_sine(self,
                         off: float = 0.5,
                         end: float = 1.0,
                         power: float = 1.0,
                         first_point_scale: float = 1.0,
                         start: int = 0, size: t.Optional[int] = None,
                         update_meta: bool = True) -> None:
        super().apodization_sine(off=off, end=end, power=power,
                                 first_point_scale=first_point_scale,
                                 start=start, size=size,
                                 update_meta=update_meta)

        # Update the metadata values
        if update_meta:
            dim = self.order[-1]
            new_apod_code = find_mapping('apodization',
                                         ApodizationType.SINEBELL,
                                         reverse=True)
            self.meta[f"FDF{dim}APODCODE"] = float(new_apod_code)
            self.meta[f"FDF{dim}APODQ1"] = float(off)
            self.meta[f"FDF{dim}APODQ2"] = float(end)
            self.meta[f"FDF{dim}APODQ3"] = float(power)

            # Update other meta dict values
            self.update_meta()

    def extract(self,
                start: t.Union[int, float],
                unit_start: UnitType,
                end: t.Union[int, float],
                unit_end: UnitType,
                update_meta: bool = True):
        old_pts = self.data.size()[-1]  # Old data size
        start, end = super().extract(start=start, unit_start=unit_start,
                                     end=end, unit_end=unit_end,
                                     update_meta=update_meta)
        new_pts = (end - start)  # New data size

        # Update the meta dict
        if update_meta and self.domain_type[-1]:
            # Get the last (current) dimension
            dim = self.order[-1]

            # Update FDFnCENTER
            if self.domain_type[-1] is DomainType.TIME:
                new_size = self.data.size()[-1]
                if self.data_type[-1] is DataType.COMPLEX:
                    center = float(round(1. + new_size / 2.))
                elif self.data_type[-1] is DataType.REAL:
                    center = float(round(1. + new_size / 2.))
                else:
                    raise NotImplementedError
            elif self.domain_type[-1] is DomainType.FREQ:
                center = self.meta[f"FDF{dim}CENTER"] - start
            self.meta[f"FDF{dim}CENTER"] = center

            # Update FDFnAPOD
            self.meta[f"FDF{dim}APOD"] = float(end - start)

            # Update FDSIZE
            self.meta[f"FDSIZE"] = float(end - start)

            # Update time-domain size (FDFnTDSIZE)
            if self.domain_type[-1] is DomainType.TIME:
                self.meta[f"FDF{dim}TDSIZE"] = float(end - start)

            # Update FDFnSW. This is only done by NMRPipe for the frequency
            # domain
            if self.domain_type[-1] is DomainType.FREQ:
                self.meta[f"FDF{dim}SW"] = self.sw_hz[-1] * new_pts / old_pts

            # Update the FDFnX1 and FDFnXN extracted ranges. This is only
            # done by NMRPipe when the dimension is in the frequency domain
            if self.domain_type[-1] is DomainType.FREQ:
                self.meta[f"FDF{dim}X1"] = float(start + 1)
                self.meta[f"FDF{dim}XN"] = float(end)

            # Update the ORIG frequency, which the frequency of the last
            # point.
            # ORIG = ORIG_OLD + df_hz * (old_end_pt - new_end_pt)
            # This is only done by NMRPipe for the frequency domain
            if self.domain_type[-1] is DomainType.FREQ:
                orig = self.meta[f"FDF{dim}ORIG"]
                orig += ((self.sw_hz[-1] / new_pts) * (old_pts - end))
            elif self.domain_type[-1] is DomainType.TIME:
                orig = (self.car_hz[-1]
                        - (self.sw_hz[-1] / new_pts) * (new_pts - center))
            else:
                raise NotImplementedError
            self.meta[f"FDF{dim}ORIG"] = orig

            # Update other meta values
            self.update_meta()

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
        # Setup the arguments
        if auto:
            if self.domain_type[-1] == DomainType.FREQ:
                # The current dimension is in the freq domain
                inv = False  # inverse transform, reverse in NMRPipe
                real = False  # do not perform a real FT
                alt = False  # do not perform sign alternation
                neg = False  # do not perform negation of imaginaries
            else:
                # The current dimension is in the time domain
                inv = True  # forward transform, reversed in NMRPipe

                # Real, TPPI and Sequential data is real transform
                # TODO: Evaluation of this flag differs from NMRPipe/nmrglue
                real = self.plane2dphase in (Plane2DPhase.MAGNITUDE,
                                             Plane2DPhase.TPPI)

                # Alternate sign, based on sign_adjustment
                # TODO: The commented out section differs from NMRPipe/nmrglue
                alt = self.sign_adjustment[-1] in (
                    SignAdjustment.REAL,
                    SignAdjustment.COMPLEX,
                    # SignAdjustment.NEGATE_IMAG,
                    # SignAdjustment.REAL_NEGATE_IMAG,
                    # SignAdjustment.COMPLEX_NEGATE_IMAG
                    )

                neg = self.sign_adjustment[-1] in (
                    SignAdjustment.NEGATE_IMAG,
                    SignAdjustment.REAL_NEGATE_IMAG,
                    SignAdjustment.COMPLEX_NEGATE_IMAG)

        # Conduct the Fourier transform
        rv = super().ft(auto=False, real=real, inv=inv, alt=alt, neg=neg,
                        bruk=bruk)

        # Update the metadata dict as needed
        if update_meta:
            dim = self.order[-1]
            new_sign_adjustment = find_mapping('sign_adjustment',
                                               SignAdjustment.NONE,
                                               reverse=True)
            self.meta[f'FDF{dim}AQSIGN'] = new_sign_adjustment

            # Switch the domain type, based on the type of Fourier Transform
            new_domain_type = DomainType.TIME if inv else DomainType.FREQ
            new_domain_type = find_mapping('domain_type', new_domain_type,
                                           reverse=True)
            self.meta[f"FDF{dim}FTFLAG"] = new_domain_type

            # Update the FTSIZE
            self.meta[f"FDF{dim}FTSIZE"] = float(self.data.size()[-1])

            # Switch the DMX ON flag to indicate that the digital filter was
            # corrected
            if self.correct_digital_filter:
                self.meta[f"FDDMXFLAG"] = 1.0  # DMX ON

        return rv

    def phase(self, p0: float, p1: float,
              discard_imaginaries: bool = True,
              update_meta: bool = True):
        rv = super().phase(p0, p1, discard_imaginaries)

        # Update the metadata, as needed
        if update_meta:
            dim = self.order[-1]

            # Switch the quadrature (data) type from complex to real if
            # imaginaries are discarded
            if discard_imaginaries and self.data_type[-1] is DataType.COMPLEX:
                self.meta[f"FDF{dim}QUADFLAG"] = find_mapping('data_type',
                                                              DataType.REAL,
                                                              reverse=True)
            # Update the phase values
            self.meta[f"FDF{dim}P0"] = p0
            self.meta[f"FDF{dim}P1"] = p1

            # Update other meta dict values
            self.update_meta()

        return rv

    def transpose(self, dim0, dim1, interleave_complex=True,
                  update_meta: bool = True):
        # Get the mapping between the dimension order (0, 1, .. self.ndims)
        # and the F1/F2/F3/F4 dimensions
        # The order must be reversed because tensors are stored
        # outer2-outer1-inner whereas NMRPipe orders them as inner-outer1-outer2
        new_order = list(self.order)[::-1]
        new_order[dim0], new_order[dim1] = new_order[dim1], new_order[dim0]

        # Conduct the permute operation
        super().transpose(dim0, dim1, interleave_complex)

        # Update the metadata values with the new order
        if update_meta:
            for i, ord in enumerate(new_order, 1):
                self.meta[f'FDDIMORDER{i}'] = float(ord)

            # Update the number of points (size) of the direct (inner) and
            # indirect (outer) dimensions
            self.meta['FDSIZE'] = float(self.data.size()[-1])
            self.meta['FDSPECNUM'] = float(reduce(lambda x, y: x * y,
                                                  self.data.size()[:-1]))
            self.meta['FDSLICECOUNT0'] = self.meta['FDSPECNUM']

            # Set the flag to indicate that the data was transposed
            self.meta['FDTRANSPOSED'] = (0.0 if self.meta['FDTRANSPOSED'] == 1.0
                                         else 1.0)

            # NMRPipe doesn't update FDMIN/FDMAX by default for TP data, but
            # instead it invalidates the FDMIN/FDMAX values with the FDSCALEFLAG
            self.meta['FDSCALEFLAG'] = 0.0  # FDMIN/FDMAX not valid

            # Update the current (last) dimension's quadrature flag if it has
            # switched between complex <-> real
            self.meta['FDQUADFLAG'] = find_mapping('data_type',
                                                   self.data_type[-1],
                                                   reverse=True)

    def zerofill(self,
                 double: t.Optional[int] = 1,
                 double_base2: t.Optional[int] = None,
                 size: t.Optional[int] = None,
                 pad: t.Optional[int] = None,
                 update_meta: bool = True) -> None:
        # Conduct the zero fill
        super().zerofill(double=double, double_base2=double_base2, size=size,
                         pad=pad)

        # Update the metadata values with the new order
        if update_meta:
            # Get the last dimension, for which values must be updated
            dim = self.order[-1]
            new_size = self.data.size()[-1]  # New size of dim

            # Set the number of ZF points, this number is -ve for zero-fill
            self.meta[f"FDF{dim}ZF"] = -1. * float(new_size)

            # The point position for the center point.
            # (From bruk2pipe.c)
            if self.data_type[-1] is DataType.COMPLEX:
                center_pt = float(round(1. + new_size / 2.))
                freq_size = float(new_size)
            elif self.data_type[-1] is DataType.REAL:
                center_pt = float(round(1. + new_size / 4.))
                freq_size = float(new_size / 2.)
            else:
                raise NotImplementedError

            self.meta[f"FDF{dim}CENTER"] = center_pt

            # Update the ORIG frequency, which the of the last point
            # According to bruk2pipe.c:
            # xOrig = xObs*xCar - xSW*(xFreqSize - xMid)/xFreqSize;
            #       = (center_hz) - df_hz * (end_pt - center_pt)
            orig = (self.obs_mhz[-1] * self.car_ppm[-1] -  # carrier in Hz
                    self.sw_hz[-1] * (freq_size - center_pt) / freq_size)
            self.meta[f"FDF{dim}ORIG"] = orig

            # Update other meta dict values
            self.update_meta()
