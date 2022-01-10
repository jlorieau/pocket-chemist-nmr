import logging

import click

logger = logging.getLogger('pocketchemist-nmr.cli.cli')


@click.command()
@click.option('-fn',
              type=click.Choice(('SOL', 'SP', 'ZF', 'FT', 'PS'),
                                 case_sensitive=False),
              default=None,
              help="The processing function to apply")
@click.pass_context
def nmrpipe(ctx: click.Context, fn: str = None):
    """A drop-in replacement for nmrPipe"""
    # Show logging information
    logger.debug(f"nmrpipe context: {fn}")

    # Forward the command, as needed, to the processing functions
    ctx.params.pop('fn')
    if fn == 'SOL':
        ctx.forward(nmrpipe_fn_sol)



@click.command()
@click.option('-mode',
              type=click.Choice((1, 2, 3)),
              default=1,
              help="Filter mode: 1 = Low Pass, 2 = Spline, 3 = Polynomial")
def nmrpipe_fn_sol(mode):
    """Apply solvent suppression"""
    # Show logging information
    # Show logging information
    logger.debug(f"nmrpipe_fn_sol context: {mode}")
