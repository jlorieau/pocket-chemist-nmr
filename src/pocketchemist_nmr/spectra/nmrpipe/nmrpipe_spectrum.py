"""
NMRSpectrum in NMRPipe format
"""
import re
import typing as t

import numpy as np

from .constants import Plane2DPhase, SignAdjustment, find_mapping
from .fileio import load_nmrpipe_tensor, load_nmrpipe_multifile_tensor
from ..nmr_spectrum import NMRSpectrum
from ..constants import DomainType

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
    """
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
        return tuple(fddimorder[:self.ndims])

    @property
    def domain_type(self) -> t.Tuple[DomainType, ...]:
        # Setup mappings between DomainTypes and the meta dict values
        domain_types = []
        for dim in self.order:
            value = self.meta[f"FDF{dim}FTFLAG"]
            domain_types.append(find_mapping('domain_type', value))
        return tuple(domain_types)

    @property
    def sw(self):
        """Spectral widths (in Hz) of all available dimensions, as ordered by
        self.order"""
        return tuple(self.meta[f"FDF{dim}SW"] for dim in self.order)

    @property
    def label(self) -> t.Tuple[str, ...]:
        return tuple(self.meta[f"FDF{dim}LABEL"] for dim in self.order)

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
             in_filepath: t.Optional['pathlib.Path'] = None,
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
        # # Setup arguments
        # out_filepath = (out_filepath if out_filepath is not None else
        #                 self.out_filepath)
        # dic = self.meta
        #
        # # Save the spectrum
        # if isinstance(self.iterator, pipe_fileio.iter3D):
        #     # The data must be convert to float32 for the following write
        #     # function
        #     data = self.data.view(np.float32)
        #     self.iterator.write(filemask=str(out_filepath), plane=data,
        #                         dic=dic)
        # else:
        #     pipe_fileio.write(filename=str(out_filepath), dic=dic,
        #                       data=self.data, overwrite=overwrite)

    # Manipulator methods

    def permute(self, new_dims: t.Tuple[int, ...]):
        # Get the old dimension order axis -> F1/F2/F3
        old_order = {i: order for i, order in enumerate(self.order)}
        super().permute(new_dims)

        # Change the dimension ordering in the metadata
        for i, new_dim in enumerate(new_dims):
            self.meta[f'FDDIMORDER{i}'] = old_order[i]

    def ft(self,
           auto: bool = False,
           real: bool = False,
           inv: bool = False,
           alt: bool = False,
           neg: bool = False,
           bruk: bool = False,
           data: t.Optional['numpy.ndarray'] = None,
           **kwargs):
        # Setup the arguments
        if auto:
            if self.is_freq():
                # The current dimension is in the freq domain
                inv = True  # perform inverse FT
                real = False  # do not perform a real FT
                alt = False  # do not perform sign alternation
                neg = False  # do not perform negation of imaginaries
            else:
                # The current dimension is in the time domain
                inv = False  # do not perform inverse FT

                # Real, TPPI and Sequential data is real transform
                # TODO: Evaluation of this flag differs from NMRPipe/nmrglue
                real = self.plane2dphase in (Plane2DPhase.MAGNITUDE,
                                             Plane2DPhase.TPPI)

                # Alternate sign, based on sign_adjustment
                # TODO: The commented out section differs from NMRPipe/nmrglue
                alt = self.sign_adjustment() in (
                    SignAdjustment.REAL,
                    SignAdjustment.COMPLEX,
                    # SignAdjustment.NEGATE_IMAG,
                    # SignAdjustment.REAL_NEGATE_IMAG,
                    # SignAdjustment.COMPLEX_NEGATE_IMAG
                    )

                neg = self.sign_adjustment() in (
                    SignAdjustment.NEGATE_IMAG,
                    SignAdjustment.REAL_NEGATE_IMAG,
                    SignAdjustment.COMPLEX_NEGATE_IMAG)

        # Update the self.meta dict as needed
        self.sign_adjustment(value=SignAdjustment.NONE)  # sign adj. applied

        # Switch the domain type, based on the type of Fourier Transform
        new_domain_type = DomainType.TIME if inv else DomainType.FREQ
        self.domain_type(value=new_domain_type)

        # Switch the inv flag. This is because NMRPipe, by default, uses ifft
        # to describe fft (i.e. positive frequencies are on the left, negative
        # frequencies are on the right)
        inv = not inv

        return super().ft(ft_func=ft_func, auto=False, real=real, inv=inv,
                          alt=alt, neg=neg, bruk=bruk, data=data)
