import logging
import typing as t
import sys
import pickle

import click
from click_option_group import optgroup, MutuallyExclusiveOptionGroup
from humanfriendly.tables import format_pretty_table

from ..processors import fileio
from ..processors import processor

logger = logging.getLogger('pocketchemist-nmr.cli.cli')


# Core plugin functionality

# Allow both '--help' and '-help' options to match nmrPipe interface
CONTEXT_SETTINGS = dict(help_option_names=['-help', '--help'])


def write_stdout(processor):
    """A function to encode processor(s) for output to the stdout.

    This function is used for transferring processors with pipes.
    """
    pickle.dump(processor, sys.stdout.buffer)


def read_stdin(cls_type=processor.NMRGroupProcessor):
    """A function to load processor(s) from input of the stdin.

    This function is used for transferring processors with pipes.
    """
    processor = pickle.load(sys.stdin.buffer)
    if cls_type is not None:
        assert isinstance(processor, cls_type)
    return processor


class HyphenGroup(click.Group):
    """A command group that handles group commands that start with hyphens"""

    # The name of commands and groups whose preceeding hyphen should be
    # stripped to allow proper routing.
    hyphen_groups = ('-fn', '-in', '-out')

    def parse_args(self, ctx: click.Context, args: t.List[str]) -> t.List[str]:
        """Parse group arguments to hyphen groups"""
        # Convert '-fn' to 'fn'
        args = [arg.lstrip('-') if arg in self.hyphen_groups
                else arg for arg in args]
        return super().parse_args(ctx, args)

    def get_command(self, ctx: click.Context, cmd_name: str) \
            -> t.Optional[click.Command]:
        """Retrieve commands and groups that are named with hyphens"""
        # Add the hyphen back to the command name, if needed
        if '-' + cmd_name in self.hyphen_groups:
            return super().get_command(ctx, '-' + cmd_name)
        else:
            return super().get_command(ctx, cmd_name)


@click.group(cls=HyphenGroup)
@click.pass_context
def nmrpipe(ctx: click.Context):
    """A drop-in replacement for nmrPipe"""
    pass


# Spectrum input

@nmrpipe.command(name='-in', context_settings=CONTEXT_SETTINGS)
@click.option('-fmt', '--format',
              type=click.Choice(('nmrpipe',)),
              default='nmrpipe', show_default=True,
              help='The format of the loaded spectrum')
@click.option('-hdr', '--show-header',
              is_flag=True, default=False,
              help="Output information on the spectrum's header")
@click.argument('in_filepaths', nargs=-1)
def nmrpipe_in(format, show_header, in_filepaths):
    """NMR spectra to load in"""
    logging.debug(f"nmrpipe_in: in_filepaths={in_filepaths}")

    # Setup a Group processor and a processor to load spectra
    group = processor.NMRGroupProcessor()
    group += fileio.LoadSpectra(in_filepaths=in_filepaths, format=format)

    # Write the objects to stdout
    if show_header:
        # Load the spectrum
        rv = group.process()
        assert 'spectra' in rv
        spectra = rv['spectra']

        for table_number, spectrum in enumerate(spectra, 1):
            click.echo(click.style(f"Table {table_number}. ", bold=True) +
                       f"Spectrum parameter for '{spectrum.in_filepath}'.")
            click.echo(format_pretty_table(sorted(spectrum.meta.items()),
                                           ('Name', 'Value')))
    else:
        write_stdout(group)


@nmrpipe.command(name='-out', context_settings=CONTEXT_SETTINGS)
@click.option('-fmt', '--format',
              type=click.Choice(('default',)),
              default='default', show_default=True,
              help='The format of the loaded spectrum')
@click.option('-ov', '--overwrite', is_flag=True,
              default=True, show_default=True)
@click.argument('out_filepaths', nargs=-1)
def nmrpipe_out(format, overwrite, out_filepaths):
    """The NMR spectra to save"""
    logging.debug(f"nmrpipe_out: out_filepaths={out_filepaths}")

    # Unpack the stdin
    group = read_stdin()

    # Setup a Group processor and a processor to load spectra
    group += fileio.SaveSpectra(out_filepaths=out_filepaths, format=format,
                                overwrite=overwrite)

    # Run the processor group
    kwargs = group.process()


# Spectrum processing functions

@nmrpipe.group(name='-fn', context_settings=CONTEXT_SETTINGS)
def nmrpipe_fn():
    """A processing function for a spectrum"""
    pass


@nmrpipe_fn.command(name='SOL', context_settings=CONTEXT_SETTINGS)
@click.option('-mode', type=click.Choice(('1', '2', '3')),
              default='1', show_default=True,
              help="Filter Mode: 1 = Low Pass, 2 = Spline, 3 = Polynomial")
@click.option('-fl', type=int, default=16, show_default=True,
              help="Filter length (low pass filter)")
@click.option('-fs', type=int, default=1, show_default=True,
              help="Filter shape: 1 = Boxcar, 2 = Sine, 3 = Sine^2 "
                   "(low pass filter)")
def nmrpipe_fn_sol(mode, fl, fs):
    """Solvent suppression"""
    # Unpack the stdin
    group = read_stdin()

    # Write the objects to stdout
    write_stdout(group)


@nmrpipe_fn.command(name='FT', context_settings=CONTEXT_SETTINGS)

@optgroup.group("Fourier Transform mode",
                help="The type of Fourier Transformation to conduct",
                cls=MutuallyExclusiveOptionGroup)
@optgroup.option('-auto', 'mode', flag_value='auto', type=click.STRING,
                 default=True, show_default=True,
                 help='Choose FT mode automatically')
@optgroup.option('-real', 'mode', flag_value='real', type=click.STRING,
                 help='Transform real data only')
@optgroup.option('-inv', 'mode', flag_value='inv', type=click.STRING,
                 help='Perform an inverse transform')

# @optgroup.group("Fourier Transform options",
#                 help="Optional processing methods for the Fourier Transform")
# @optgroup.option('-alt', is_flag=True,
#                  help="Apply sign alternation")
# @optgroup.option('-neg', is_flag=True,
#                  help="Negate the imaginary component(s)")

#@click.argument('stdin', default=sys.stdin)
def nmrpipe_fn_ft(mode):
    """Complex Fourier Transform"""
    logging.debug(f"nmrpipe_fn_ft: mode={mode}")

    # Unpack the stdin
    group = read_stdin()

    # Add the FT processor
    group += processor.FTSpectra(mode=mode)

    # Write the objects to stdout
    write_stdout(group)
