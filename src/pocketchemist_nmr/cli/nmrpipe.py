"""Click interface for nmrpipe commands"""

import io
import typing as t
import sys
import pickle
from contextlib import redirect_stdout

import click
from click_option_group import optgroup, MutuallyExclusiveOptionGroup
from loguru import logger
from humanfriendly.tables import format_pretty_table

from ..spectra.constants import UnitType


# Core plugin functionality

# Allow both '--help' and '-help' options to match nmrPipe interface
CONTEXT_SETTINGS = dict(help_option_names=['-help', '--help'])


def write_stdout(processor):
    """A function to encode processor(s) for output to the stdout.

    This function is used for transferring processors with pipes.
    """
    pickle.dump(processor, sys.stdout.buffer)


def read_stdin():
    """A function to load processor(s) from input of the stdin.

    This function is used for transferring processors with pipes.
    """
    processor = pickle.load(sys.stdin.buffer)
    return processor


class HyphenGroup(click.Group):
    """A command group that handles group commands that start with hyphens"""

    # The name of commands and groups whose preceeding hyphen should be
    # stripped to allow proper routing.
    hyphen_groups = ('-fn', '-in')

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


# Spectrum input/output

def nmrpipe_out(func):
    @click.option('-out', '--out-filepaths', default=None,
                  help="Filename to write spectrum")
    @click.option('-outfmt', '--out-format',
                  type=click.Choice(('default',)),
                  default='default', show_default=True,
                  help='The format of the saved spectrum')
    @click.option('-ov', '--overwrite', is_flag=True, default=True,
                  show_default=True,
                  help="Overwrite the file if it exists")
    def _nmrpipe_out(out_filepaths, out_format, overwrite, *args, **kwargs):
        logger.debug(f"out_filepaths={out_filepaths}")

        # Run the inner function, capturing the stdout
        fd = io.BytesIO()
        buff = io.TextIOWrapper(fd, sys.stdout.encoding)
        with redirect_stdout(buff):
            rv = func(*args, **kwargs)

        if out_filepaths is not None:
            # If a output file was specified, write it to the disk
            from ..processors.fileio import SaveSpectra

            # Unpack the stdin
            buff.buffer.seek(io.SEEK_SET)  # Reset buffer to start
            group = pickle.load(buff.buffer)  # Read in the stdout

            # Setup a Group processor and a processor to load spectra
            group += SaveSpectra(out_filepaths=out_filepaths, format=out_format,
                                 overwrite=overwrite)

            # Run the processor group
            kwargs = group.process()
        else:
            # Otherwise write it to stdout as usual
            buff.buffer.seek(io.SEEK_SET)  # Reset buffer to start
            sys.stdout.buffer.write(buff.buffer.read())  # Send buffer to stdout

        return rv
    _nmrpipe_out.__doc__ = func.__doc__  # pass the docstring to wrapper
    return _nmrpipe_out


@nmrpipe.command(name='-in', context_settings=CONTEXT_SETTINGS)
@click.option('-infmt', '--in-format',
              type=click.Choice(('nmrpipe',)),
              default='nmrpipe', show_default=True,
              help='The format of the loaded spectrum')
@click.option('-hdr', '--show-header',
              is_flag=True, default=False,
              help="Output information on the spectrum's header")
@click.argument('in_filepaths', nargs=-1)
@nmrpipe_out
def nmrpipe_in(in_format, show_header, in_filepaths):
    """NMR spectra to load in"""
    from ..processors.processor import NMRGroupProcessor
    from ..processors.fileio import LoadSpectra

    logger.debug(f"in_filepaths={in_filepaths}")

    # Setup a Group processor and a processor to load spectra
    group = NMRGroupProcessor()
    group += LoadSpectra(in_filepaths=in_filepaths, format=in_format)

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


# Spectrum processing functions
# These are ordered the same way as listed in NMRPipe

@nmrpipe.group(name='-fn', context_settings=CONTEXT_SETTINGS)
def nmrpipe_fn():
    """A processing function for a spectrum"""
    pass


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
@nmrpipe_out
# @optgroup.group("Fourier Transform options",
#                 help="Optional processing methods for the Fourier Transform")
# @optgroup.option('-alt', is_flag=True,
#                  help="Apply sign alternation")
# @optgroup.option('-neg', is_flag=True,
#                  help="Negate the imaginary component(s)")
def nmrpipe_fn_ft(mode):
    """Complex Fourier Transform"""
    from ..processors.processor import FTSpectra
    logger.debug(f"mode={mode}")

    group = read_stdin()  # Unpack the stdin
    group += FTSpectra(mode=mode)  # Add the FT processor
    write_stdout(group)  # Write the objects to stdout


@nmrpipe_fn.command(name='TP', context_settings=CONTEXT_SETTINGS)
@nmrpipe_out
def nmrpipe_fn_tp():
    """Transpose the last 2 dimensions of the spectrum (XY -> YX)"""
    from ..processors.processor import Transpose2D

    group = read_stdin()  # Unpack the stdin
    group += Transpose2D()  # Add the FT processor
    write_stdout(group)  # Write the objects to stdout


@nmrpipe_fn.command(name='ZF', context_settings=CONTEXT_SETTINGS)
@click.option('-zf', required=False, type=float, default=1.0,
              help="Number of times to double the size with zero-filling")
@click.option('-zf2', required=False, type=float, default=1.0,
              help="Number of times to double the size to match the next 2^N "
                   "size with zero-filling")
@click.option('-pad', required=False, type=float, default=None,
              help="Number of points to add with zero-filling")
@click.option('-size', required=False, type=float, default=None,
              help="Final size after zero-filling")
@nmrpipe_out
def nmrpipe_fn_zf(zf, zf2, pad, size):
    """Zero-fill the last dimension of a spectrum"""
    from ..processors.processor import ZeroFillSpectra
    logger.debug(f"zf={zf}, zf2={zf2}, pad={pad}, size={size}")

    group = read_stdin()  # Unpack the stdin
    group += ZeroFillSpectra(double=zf, double_base2=zf2, size=size,
                             pad=pad)  # Add processor
    write_stdout(group)  # Write the objects to stdout


@nmrpipe_fn.command(name='PS', context_settings=CONTEXT_SETTINGS)
@click.option('-p0', required=False, type=float, default=0.0,
              help="The zeroth order (frequency independent) phase correction")
@click.option('-p1', required=False, type=float, default=0.0,
              help="The first order (linear frequency) phase correction")
@click.option('-di', is_flag=True, type=bool, default=False, show_default=True,
              help="Discard imaginary component")
@nmrpipe_out
def nmrpipe_fn_ps(p0, p1, di):
    """Phase the last dimension of a spectrum"""
    from ..processors.processor import PhaseSpectra
    logger.debug(f"p0={p0}, p1={p1}, di={di}")

    group = read_stdin()  # Unpack the stdin
    group += PhaseSpectra(p0=p0, p1=p1, discard_imaginaries=di)  # Add processor
    write_stdout(group)  # Write the objects to stdout


@nmrpipe_fn.command(name='EM', context_settings=CONTEXT_SETTINGS)
@click.option('-lb', required=True, type=float,
              help="Exponential broadening rate (Hz)")
@click.option('-start', required=False, type=int, default=0,
              help="First point to start apodization")
@click.option('-size', required=False, type=int, default=None,
              help="Number of points to apodize")
@nmrpipe_out
def nmrpipe_fn_em(lb, start, size):
    """Apodize the last dimension with exponential multiplication (Lorentzian)
    """
    from ..processors.processor import ApodizationExpSpectra
    logger.debug(f"lb={lb}, start={start}, size={size}")

    group = read_stdin()
    group += ApodizationExpSpectra(lb=lb, start=start, size=size)
    write_stdout(group)


@nmrpipe_fn.command(name='SP', context_settings=CONTEXT_SETTINGS)
@click.option('-off', required=False, type=float, default=0.5,
              help="Offset of the sine-bell in units of pi")
@click.option('-end', required=False, type=float, default=1.0,
              help="The end of the sine-bell in units of pi")
@click.option('-pow', required=False, type=float, default=1.0,
              help="Exponent of the sine-bell")
@click.option('-start', required=False, type=int, default=0,
              help="First point to start apodization")
@click.option('-size', required=False, type=int, default=None,
              help="Number of points to apodize")
@nmrpipe_out
def nmrpipe_fn_sp(off, end, pow, start, size):
    """Apodize the last dimension with a sinebell power function"""
    from ..processors.processor import ApodizationSinebellSpectra
    logger.debug(f"off={off}, end={end}, pow={pow}, start={start}, size={size}")

    group = read_stdin()
    group += ApodizationSinebellSpectra(off=off, end=end, power=pow,
                                        start=start, size=size)
    write_stdout(group)


@nmrpipe_fn.command(name='EXT', context_settings=CONTEXT_SETTINGS)
@click.option('-x1', '--start', required=False, type=str, default='0',
              help="Region range start (no units in points, Hz, sec, %, or "
                   "PPM)")
@click.option('-xn', '--end', required=False, type=str, default='NPTS',
              help="Region range end (no units in points, Hz, sec, %, or "
                   "PPM). Negative point values count from the end of the "
                   "dimension.")
@click.option('-sw', is_flag=True, type=bool, default=True, show_default=True,
              help="Update the spectral width and position parameters")
@nmrpipe_out
def nmrpipe_fn_ext(start, end, sw):
    """Extract a region from the last dimension"""
    from ..processors.processor import ExtractSpectra

    # Convert x1/xn
    start, unit_start = UnitType.from_string(start)
    if end == 'NPTS':
        end, unit_end = -1, UnitType.POINTS
    else:
        end, unit_end = UnitType.from_string(end)

    logger.debug(f"start={start} {unit_start}, end={end} {unit_end}, sw={sw}")

    group = read_stdin()
    group += ExtractSpectra(start=start, unit_start=unit_start,
                            end=end, unit_end=unit_end, update_meta=sw)
    write_stdout(group)
