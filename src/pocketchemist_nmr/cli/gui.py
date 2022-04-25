"""Click interface for NMRDesk gui"""
import click
from PyQt6.QtWidgets import QApplication

from ..gui import NMRDeskWindow


@click.command
@click.argument('args', nargs=-1)
def nmrdesk(args):
    """The NMRDesk graphical user interface (GUI)"""
    # Create the root app
    app = QApplication(list(args))

    # Create the main window
    window = NMRDeskWindow()
    window.show()

    # Start root app
    app.exec()
