"""
Test the nmr plugin CLI functionality
"""
import pytest
from click.testing import CliRunner
from pytest_cases import parametrize_with_cases
from pocketchemist.cli import get_root_command


def test_cli_nmrpipe(runner=CliRunner()):
    """Test the loading of the nmrpipe plugin"""
    result = runner.invoke(get_root_command(), ['nmrpipe', '--help'])
    assert result.exit_code == 0  # Command successfully executed


def test_cli_nmrpipe_in(runner=CliRunner()):
    """Test the nmrpipe plugin '-in' option."""
    result = runner.invoke(get_root_command(), ['nmrpipe', '-in'])
    assert result.exit_code == 0  # Command successfully executed


@pytest.mark.parametrize('fn, opts',
                         (('EM', ('-lb', '10.')),
                          ('EXT', ()),
                          ('FT', ()),
                          ('PS', ()),
                          ('SP', ()),
                          ('TP', ()),
                          ('ZF', ()),
                          ))
@parametrize_with_cases('dataset', cases="..cases.nmrpipe", prefix='data',
                        glob='data_nmrpipe_complex_fid_2d')
def test_cli_nmrpipe_fn(fn, opts, dataset, tmpdir, runner=CliRunner()):
    """Test the nmrpipe subcommand processor functions."""
    filepath = dataset['filepath']

    # Run the command to load the file
    args = ('nmrpipe', '-in', str(filepath))
    result = runner.invoke(get_root_command(), args)
    assert result.exit_code == 0

    # Run the processing fn command and output to a tmpdir
    args = ('nmrpipe', '-fn', fn, *opts, '-out', tmpdir / filepath.name)
    result = runner.invoke(get_root_command(), args, input=result.stdout_bytes)

    if result.exception is not None:
        raise result.exception

    assert result.exit_code == 0
    assert (tmpdir / filepath.name).exists()
