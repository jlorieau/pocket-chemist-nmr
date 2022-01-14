"""
NMRSpectrum in NMRPipe format
"""
import re

from .nmr_spectrum import NMRSpectrum

import nmrglue as ng
import typing as t

__all__ = ('NMRPipeSpectrum',)


class NMRPipeSpectrum(NMRSpectrum):
    """An NMRpipe spectrum"""

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
             force_multifile: bool = False,
             in_plane: str = 'x', out_plane: str = 'DEFAULT'):
        """Load the NMRPipeSpectrum.

        Parameters
        ----------
        in_filepath
            The filepath for the spectrum file(s) to load.
        force_multifile
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
        is_multifile = re.search(r'%\d+d', str(self.in_filepath)) is not None

        # Load the spectrum
        if is_multifile or force_multifile:
            # Load the iterator
            data = ng.pipe.iter3D(str(self.in_filepath), in_plane, out_plane)

            # Get the first dict to populate self.meta
            dic, _ = data.next()
            data.reinitialize()

            self.meta = dic
            self.data = data

            # TODO: Implement assignment of self.meta
            self.is_multifile = True
        else:
            dic, data = ng.pipe.read(str(self.in_filepath))
            self.meta = dic
            self.data = data
            self.is_multifile = False

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
        out_filepath = (out_filepath if out_filepath is not None else
                        self.out_filepath)

        if self.is_multifile:
            raise NotImplementedError
        else:
            dic = self.meta
            ng.pipe.write(out_filepath, dic, self.data, overwrite=overwrite)
