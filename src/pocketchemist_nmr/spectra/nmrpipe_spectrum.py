"""
NMRSpectrum in NMRPipe format
"""
import re
import typing as t

import numpy as np
import scipy.fft as fft
import nmrglue as ng

from .nmr_spectrum import NMRSpectrum

__all__ = ('NMRPipeSpectrum',)


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

    @property
    def ndims(self):
        if 'FDDIMCOUNT' in self.meta:
            return int(self.meta['FDDIMCOUNT'])
        else:
            return len(self.data.shape)

    @property
    def order(self) -> t.Tuple[int, ...]:
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
            iterator = ng.pipe.iter3D(str(self.in_filepath),
                                      in_plane, out_plane)

            self.iterator = iterator
        else:
            dic, data = ng.pipe.read(str(self.in_filepath))
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
        if isinstance(self.iterator, ng.pipe.iter3D):
            # The data must be convert to float32 for the following write
            # function
            data = self.data.view(np.float32)
            self.iterator.write(filemask=str(out_filepath), plane=data,
                                dic=dic)
        else:
            ng.pipe.write(filename=str(out_filepath), dic=dic, data=self.data,
                          overwrite=overwrite)

    def ft(self=None,
           mode: t.Optional[str] = 'auto',
           fft_func: t.Callable = fft.fft,
           **kwargs):
        if mode == 'auto':
            def _fft_func(data):
                def fft(data):
                    (fft.fft(fft.ifftshift(data, -1),
                             axis=-1).astype(data.dtype) * scale)
                fft_func = lambda data: (fft.fft(fft.ifftshift(data, -1)))
