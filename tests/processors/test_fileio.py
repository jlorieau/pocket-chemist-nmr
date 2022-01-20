"""Tests for File I/O NMR spectrum processors"""
from pathlib import Path

import pytest

from pocketchemist_nmr.processors.fileio import LoadSpectra, SaveSpectra
from pocketchemist_nmr.spectra import NMRPipeSpectrum

spectrum2d_nmrpipe = (Path('data') / 'bruker' /
                      'CD20170124_av500hd_101_ubq_hsqcsi2d' /
                      'hsqcetfpf3gpsi2.ft2',)

spectrum3d_nmrpipe = (Path('data') / 'bruker' /
                      'CD20170124_av500hd_103_ubq_hncosi2d' /
                      'fid' / 'test%03d.fid',)


def get_spectra():
    return ([(spec, NMRPipeSpectrum, 2) for spec in spectrum2d_nmrpipe] +
            [(spec, NMRPipeSpectrum, 3) for spec in spectrum3d_nmrpipe])


@pytest.mark.parametrize("in_filepath,spectrum_cls,ndims", get_spectra())
def test_load_spectra_nmrpipe(in_filepath, spectrum_cls, ndims):
    """Test the LoadSpectra processor"""
    # Run the processor
    if spectrum_cls == NMRPipeSpectrum:
        fmt = 'nmrpipe'
    else:
        raise NotImplementedError

    processor = LoadSpectra(in_filepaths=in_filepath, format=fmt)
    kwargs = processor.process()

    # Check that the spectrum was correctly loaded and returned in the kwargs
    assert 'spectra' in kwargs
    assert len(kwargs['spectra']) == 1

    # Check the spectrum's class and metadata
    spectrum = kwargs['spectra'][0]
    assert isinstance(spectrum, spectrum_cls)
    assert spectrum.ndims == ndims


@pytest.mark.parametrize("in_filepath,spectrum_cls,ndims", get_spectra())
def test_save_spectra_nmrpipe(in_filepath, spectrum_cls, ndims, tmpdir):
    """Test the SaveSpectra processor"""
    tmpdir = Path(tmpdir)

    # Load the spectrum
    if spectrum_cls == NMRPipeSpectrum:
        fmt = 'nmrpipe'
    else:
        raise NotImplementedError

    loader = LoadSpectra(in_filepaths=in_filepath, format=fmt)
    kwargs = loader.process()

    # Save the spectra to a new place
    out_filepath = tmpdir / in_filepath
    saver = SaveSpectra(out_filepaths=out_filepath, format=fmt)
    saver.process(**kwargs)

    # Check that the file exists
    if "%03d" in str(out_filepath):
        # This uses a filemask, check the first file
        out_filepath = Path(str(out_filepath).replace("%03d", "001"))

    assert out_filepath.exists()
