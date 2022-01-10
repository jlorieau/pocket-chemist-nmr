import logging
import typing as t

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

logger = logging.getLogger('pocketchemist-nmr.cli.cli')

class HyphenGroup(click.Group):
    """A command group that handles group commands that start with hyphens"""

    # The name of commands and groups whose preceeding hyphen should be
    # stripped to allow proper routing.
    hyphen_groups = ('-fn',)

    def parse_args(self, ctx: click.Context, args: t.List[str]) -> t.List[str]:
        """Parse group arguments by routing """
        # Convert '-fn' to 'fn'
        args = [arg.lstrip('-') if arg in self.hyphen_groups
                else arg for arg in args]
        return super().parse_args(ctx, args)

    def get_command(self, ctx: click.Context, cmd_name: str) \
            -> t.Optional[click.Command]:
        # Add the hyphen back to the command name, if needed
        if '-' + cmd_name in self.hyphen_groups:
            return super().get_command(ctx, '-' + cmd_name)
        else:
            return super().get_command(ctx, cmd_name)


@click.group(cls=HyphenGroup)
@click.pass_context
def nmrpipe(ctx: click.Context, fn: str = None):
    """A drop-in replacement for nmrPipe"""
    # Show logging information
    pass


@nmrpipe.group(name='-fn')
def nmrpipe_fn():
    """A processing function for a spectrum"""
    pass


@nmrpipe_fn.command(name='SOL')
@click.option('-mode', type=click.Choice(('1', '2', '3')),
              default='1', show_default=True,
              help="Filter Mode: 1 = Low Pass, 2 = Spline, 3 = Polynomial")
@click.option('-fl', type=int, default=16, show_default=True,
              help="Filter length (low pass filter)")
@click.option('-fs', type=int, default=1, show_default=True,
              help="Filter shape: 1 = Boxcar, 2 = Sine, 3 = Sine^2 "
                   "(low pass filter)")

def nmrpipe_fn_sol():
    """Solvent suppression"""
    pass


@nmrpipe_fn.command(name='FT')
@optgroup.group("Fourier Transform method",
                help="The type of Fourier Transformation to conduct",
                cls=RequiredMutuallyExclusiveOptionGroup)
@optgroup.option('-auto', is_flag=True, default=True, show_default=True,
                 help='Choose FT mode automatically')
@optgroup.option('-real', is_flag=True,
                 help='Transform real data only')
@optgroup.option('-inv', is_flag=True,
                 help='Perform an inverse transform')
@optgroup.group("Fourier Transform options",
                help="Optional processing methods for the Fourier Transform")
@optgroup.option('-alt', is_flag=True,
                 help="Apply sign alternation")
@optgroup.option('-neg', is_flag=True,
                 help="Negate the imaginary component(s)")
def nmrpipe_fn_ft():
    """Complex Fourier Transform"""
    pass