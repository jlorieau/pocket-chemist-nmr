"""
Test the nmr plugin CLI functionality
"""
import pytest
from click.testing import CliRunner
from pocketchemist.cli import main


def test_cli_nmrpipe(runner=CliRunner()):
    """Test the loading of the nmrpipe plugin"""
    result = runner.invoke(main, ['nmrpipe', '--help'])
    assert result.exit_code == 0  # Command successfully executed


def test_cli_nmrpipe_in(runner=CliRunner()):
    """Test the nmrpipe plugin '-in' option."""
    result = runner.invoke(main, ['nmrpipe', '-in'])
    assert result.exit_code == 0  # Command successfully executed


@pytest.mark.parametrize('opt,has_required_params',
                         (('FT', False),
                          ('PS', False),
                          ('EM', True)))
def test_cli_nmrpipe_fn(opt, has_required_params, runner=CliRunner()):
    """Test the nmrpipe plugin '-fn' option."""
    # 1. With '-help'
    result = runner.invoke(main, ['nmrpipe', '-fn', opt, '-help'])
    assert result.exit_code == 0  # Command successfully executed

    # 2. With '--help'
    result = runner.invoke(main, ['nmrpipe', '-fn', opt, '--help'])
    assert result.exit_code == 0  # Command successfully executed

    # 3. Without '-help' or '--help'. This command expects input from stdin
    #    so it will fail
    result = runner.invoke(main, ['nmrpipe', '-fn', opt])
    if has_required_params:
        assert result.exit_code == 2  # Command not executed successfully
    else:
        assert result.exit_code == 1  # Command not executed successfully
