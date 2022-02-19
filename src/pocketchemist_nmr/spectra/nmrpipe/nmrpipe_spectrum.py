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
from ..constants import DomainType, DataType, DataLayout, ApodizationType

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

    # Basic accessor/mutator methods

    @property
    def ndims(self):
        """The number of dimensions in the spectrum"""
        if 'FDDIMCOUNT' in self.meta:
            return int(self.meta['FDDIMCOUNT'])
        else:
            return len(self.data.shape)

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
    def sw(self):
        """Spectral widths (in Hz) of all available dimensions, as ordered by
        self.order"""
        return tuple(self.meta[f"FDF{dim}SW"] for dim in self.order)

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

    def apodization_exp(self,
                        lb: float,
                        first_point_scale: float = 1.0,
                        start: int = 0,
                        size: t.Optional[int] = None):
        super().apodization_exp(lb=lb, first_point_scale=first_point_scale,
                                start=start, size=size)

        # Update the metadata values
        dim = self.order[0]
        new_apod_code = find_mapping('apodization', ApodizationType.EXPONENTIAL,
                                     reverse=True)
        self.meta[f"FDF{dim}APODCODE"] = float(new_apod_code)
        self.meta[f"FDF{dim}APODQ1"] = float(lb)

    def transpose(self, dim0, dim1, interleave_complex=True):
        # Get the mapping between the dimension order (0, 1, .. self.ndims)
        # and the F1/F2/F3/F4 dimensions
        # The order must be reversed because tensors are stored
        # outer2-outer1-inner whereas NMRPipe orders them as inner-outer1-outer2
        new_order = list(self.order)[::-1]
        new_order[dim0], new_order[dim1] = new_order[dim1], new_order[dim0]

        # Conduct the permute operation
        super().transpose(dim0, dim1, interleave_complex)

        # Update the metadata values with the new order
        for i, ord in enumerate(new_order, 1):
            self.meta[f'FDDIMORDER{i}'] = round(ord, 1)

        self.meta['FDSIZE'] = self.data.size()[-1]
        self.meta['FDSPECNUM'] = reduce(lambda x, y: x * y,
                                        self.data.size()[:-1])
        self.meta['FDSLICECOUNT0'] = self.meta['FDSPECNUM']
        self.meta['FDTRANSPOSED'] = (0.0 if self.meta['FDTRANSPOSED'] == 1.0
                                     else 1.0)

    def phase(self, p0: float, p1: float, discard_imaginaries: bool = True):
        rv = super().phase(p0, p1, discard_imaginaries)

        # Update the header, as needed
        dim = self.order[-1]
        if discard_imaginaries and self.data_type[-1] is DataType.COMPLEX:
            self.meta[f"FDF{dim}QUADFLAG"] = find_mapping('data_type',
                                                          DataType.REAL,
                                                          reverse=True)
        self.meta[f"FDF{dim}p0"] = p0
        self.meta[f"FDF{dim}p1"] = p1

        return rv

    def ft(self,
           auto: bool = False,
           real: bool = False,
           inv: bool = False,
           alt: bool = False,
           neg: bool = False,
           bruk: bool = False,
           **kwargs):
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

        # Update the self.meta dict as needed
        last_dim = self.order[-1]
        new_sign_adjustment = find_mapping('sign_adjustment',
                                           SignAdjustment.NONE, reverse=True)
        self.meta[f'FDF{last_dim}AQSIGN'] = new_sign_adjustment

        # Switch the domain type, based on the type of Fourier Transform
        new_domain_type = DomainType.TIME if inv else DomainType.FREQ
        new_domain_type = find_mapping('domain_type', new_domain_type,
                                       reverse=True)
        self.meta[f"FDF{last_dim}FTFLAG"] = new_domain_type

        return rv
