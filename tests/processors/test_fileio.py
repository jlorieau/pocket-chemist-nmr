"""Tests for File I/O NMR spectrum processors"""
from pathlib import Path

from pytest_cases import parametrize_with_cases
from pocketchemist_nmr.processors.fileio import LoadSpectra, SaveSpectra
from pocketchemist_nmr.spectra import NMRPipeSpectrum


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='..cases')
def test_load_spectra_nmrpipe(expected):
    """Test the LoadSpectra processor"""
    # Run the processor
    processor = LoadSpectra(in_filepaths=expected['filepath'],
                            format='nmrpipe')
    kwargs = processor.process()

    # Check that the spectrum was correctly loaded and returned in the kwargs
    assert 'spectra' in kwargs
    assert len(kwargs['spectra']) == 1

    # Check the spectrum's class and metadata
    spectrum = kwargs['spectra'][0]
    assert isinstance(spectrum, NMRPipeSpectrum)
    assert spectrum.ndims == expected['header']['ndims']


@parametrize_with_cases('expected', glob='*nmrpipe*', prefix='data_',
                        cases='..cases.nmrpipe')
def test_load_spectra_nmrpipe(expected, tmpdir):
    """Test the LoadSpectra processor"""
    # Run the processor
    processor = LoadSpectra(in_filepaths=expected['filepath'],
                            format='nmrpipe')
    kwargs = processor.process()

    # Check that the spectrum was correctly loaded and returned in the kwargs
    assert 'spectra' in kwargs
    assert len(kwargs['spectra']) == 1
    spectrum = kwargs['spectra'][0]

    # Save spectra and try reloading
    out_filepath = Path(tmpdir) / expected['filepath'].name.replace("%", "")
    assert not out_filepath.exists()
    processor = SaveSpectra(out_filepaths=out_filepath,
                            format='nmrpipe')
    kwargs = processor.process(**kwargs)

    # Check that the file was saved
    assert out_filepath.exists()

    # Check that the saved spectrum can be reloaded
    processor = LoadSpectra(in_filepaths=out_filepath,
                            format='nmrpipe')
    kwargs = processor.process()

    # Check the spectrum's class and metadata
    spectrum = kwargs['spectra'][0]
    assert isinstance(spectrum, NMRPipeSpectrum)
    assert spectrum.ndims == expected['header']['ndims']

    # Delete the file
    out_filepath.unlink()
