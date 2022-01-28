"""
NMRSpectrum in NMRPipe format
"""
import re
import typing as t

import numpy as np

from . import pipe_fileio
from .constants import Plane2DPhase, SignAdjustment
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

    def __next__(self):
        try:
            meta, data = next(self.iterator)
        except StopIteration as exc:
            self.iterator_done = True
            raise exc

        self.meta = meta
        self.data = data
        return self

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
        """The ordering of dimensions in the data"""
        if 'FDDIMORDER' in self.meta:
            return tuple(map(int, self.meta['FDDIMORDER'][:self.ndims]))
        else:
            return tuple(range(1, self.ndims + 1))

    @order.setter
    def order(self, new_order: t.Tuple[int, ...]):
        assert 'FDDIMORDER' in self.meta
        assert len(new_order) == self.ndims, (
                "The new ordering must match the number of dimensions.")
        assert all(0 < i < 6 for i in new_order), (
                "The dimension numbers must be between 1-6.")

        for i, order in enumerate(new_order):
            self.meta['FDDIMORDER'][i] = float(order)

    def domain_type(self,
                    dim: int = None,
                    value: t.Optional[DomainType] = None) -> DomainType:
        dim = dim if dim is not None else self.order[0]
        ndims = self.ndims
        assert 0 < dim <= ndims, (
            f"The specified dimension '{dim}' must be between 1-{ndims}.")
        label = f'FDF{dim}FTFLAG'

        # Setup mappings between DomainTypes and the meta dict values
        mappings = {DomainType.FREQ: 1.0,
                    DomainType.TIME: 0.0}

        # Set the value, if specified
        if value is not None:
            self.meta[label] = mappings[value]

        # Retrieve and format the domain type
        if label in self.meta:
            value = self.meta[label]
            return tuple(k for k, v in mappings.items() if v == value)[0]
        else:
            return DomainType.UNKNOWN

    @property
    def sw(self):
        """Spectral widths (in Hz) of all available dimensions, as ordered by
        self.order"""
        ordered_dims = self.order[:self.ndims]
        return tuple(self.meta[f"FDF{dim}SW"] for dim in ordered_dims)

    @sw.setter
    def sw(self, value):
        # Make sure the given 'value' iterable matches the number of dimensions
        ndims = self.ndims
        ordered_dims = self.order[:ndims]
        assert len(value) == ndims, (
            f"The number of spectral widths '{value}' should match the number "
            f"of dimensions ({ndims}).")

        # Set the new spectral widths
        for dim, v in zip(ordered_dims, value):
            self.meta[f"FDF{dim}SW"] = v

    def sign_adjustment(self,
                        dim: int = None,
                        value: t.Optional[SignAdjustment] = None) \
            -> SignAdjustment:
        """Whether the dimension requires sign alternation.

        Parameters
        ----------
        dim
            The dimension (1-4) to evaluate whether it's in the frequency
             domain. By default, this returns the current dimension.
        value
            If specified, set the sign adjustment to this value

        Returns
        -------
        sign_adjustment
            The current value of the sign adjustment setting.
        """
        dim = dim if dim is not None else self.order[0]
        ndims = self.ndims
        assert 0 < dim <= ndims, (
            f"The specified dimension '{dim}' must be between 1-{ndims}.")
        label = f"FDF{dim}AQSIGN"

        if value is not None:
            self.meta[label] = value.value

        return SignAdjustment(self.meta.get(label, SignAdjustment.NONE))

    @property
    def plane2dphase(self):
        """The phase of 2D planes for 2-, 3-, 4-dimensional self.data values."""
        ndims = self.ndims
        fd2dphase = self.meta.get('FD2DPHASE', None)
        if ndims > 1 and fd2dphase is not None:
            return Plane2DPhase(fd2dphase)
        else:
            return Plane2DPhase.NONE

    @plane2dphase.setter
    def plane2dphase(self, value: Plane2DPhase):
        if value is not Plane2DPhase.NONE:
            self.meta['FD2DPHASE'] = value.value

    # I/O methods

    def load(self,
             in_filepath: t.Optional['pathlib.Path'] = None,
             force_iterator: bool = False,
             in_plane: str = 'x', out_plane: str = 'DEFAULT'):
        """Load the NMRPipeSpectrum.

        Parameters
        ----------
        in_filepath
            The filepath for the spectrum file(s) to load.
        force_iterator
            If True, force the loading of the nmrPipe spectrum using an
            iterator for multiple files, which is used for 3Ds, 4Ds, etc.
        in_plane
            The plane read as the direct dimension. e.g. 'x', 'y', 'z', 'a'
        out_plane
             The plane written as the direct dimension. e.g. 'x', 'y', 'z',
             'a', 'DEFAULT'. (Default means it's the same as the in_plane)
        """
        super().load(in_filepath=in_filepath)

        # Determine if the spectrum should be loaded as a series of planes
        # (3D, 4D, etc.) or as and 1D or 2D (plane)
        is_iterator = re.search(r'%\d+d', str(self.in_filepath)) is not None

        # Load the spectrum and assign attributes
        if is_iterator or force_iterator:
            # Load the iterator. The iterator must be run once to populate
            # self.meta and self.data (see __next__)
            iterator = pipe_fileio.iter3D(str(self.in_filepath),
                                          in_plane, out_plane)

            self.iterator = iterator
        else:
            dic, data = pipe_fileio.read(str(self.in_filepath))
            self.meta = dic
            self.data = data

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
        # Setup arguments
        out_filepath = (out_filepath if out_filepath is not None else
                        self.out_filepath)
        dic = self.meta

        # Save the spectrum
        if isinstance(self.iterator, pipe_fileio.iter3D):
            # The data must be convert to float32 for the following write
            # function
            data = self.data.view(np.float32)
            self.iterator.write(filemask=str(out_filepath), plane=data,
                                dic=dic)
        else:
            pipe_fileio.write(filename=str(out_filepath), dic=dic,
                              data=self.data, overwrite=overwrite)

    # Processing methods

    def ft(self,
           ft_func: t.Callable,
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
