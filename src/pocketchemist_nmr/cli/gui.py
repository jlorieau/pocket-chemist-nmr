"""Click interface for NMRDesk gui"""
import click
from PyQt6.QtWidgets import QApplication

from ..gui import NMRDeskWindow


@click.command()
@click.option('-p', '--nmrpipe',
              multiple=True,
              help="Filenames for spectra to open in NMRPipe format")
@click.argument('args', nargs=-1)
def nmrdesk(args, nmrpipe):
    """The NMRDesk graphical user interface (GUI)"""
    # Create the root app
    app = QApplication(list(args))

    # Set style
    app.setStyle("Fusion")

    # Create the main window
    window = NMRDeskWindow()

    # Add spectra
    for filename in nmrpipe:
        window.addSpectrum(filename)

    # Show the window and start root app
    app.exec()
