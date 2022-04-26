"""
The root window for the NMRDesk application
"""
from pathlib import Path
import typing as t

from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QMenuBar, QStatusBar,
                             QToolBar, QComboBox, QFileDialog, QMessageBox)
from PyQt6 import uic

from .plot_widgets import NMRSpectrumContour2DWidget
from ..spectra import NMRSpectrum, NMRPipeSpectrum


class NMRDeskWindow(QMainWindow):

    #: The width (in number of characters) of the plot selector combo box
    plotSelectorWidth = 60

    #: The menubar widget (top)
    menubar: QMenuBar

    #: The toolbar
    toolBar: QToolBar

    #: The statusbar widget (bottom)
    statusbar: QStatusBar

    #: A stack widget containing the different plots
    plotStack: QStackedWidget

    #: A list of opened NMR spectra
    spectra: t.List[NMRSpectrum]

    #: The selector for the plot displayed
    plotSelector: QComboBox

    def __init__(self):
        super().__init__()

        # Setup mutable containers
        self.spectra = []

        # Load designer layout and setup the window
        uic.loadUi(Path(__file__).parent / 'nmrdesk.ui', self)
        self.setupToolbar()
        self.statusbar.setVisible(False)  # Hide status bar

        self.show()

    def setupToolbar(self):
        """Setup the toolbar with extra widgets not added by designer"""
        # Add a spectrum selector combobox to the toolbar and adjust its width
        self.plotSelector = QComboBox()

        metrics = self.plotSelector.fontMetrics()
        width = metrics.boundingRect(' ' * self.plotSelectorWidth).width()
        self.plotSelector.setMinimumWidth(width)

        self.toolBar.addWidget(self.plotSelector)

        # Changes to the self.plotStack should be reflected in the combobox
        self.plotStack.currentChanged.connect(self._updatePlotSelector)
        self.plotSelector.activated.connect(self.plotStack.setCurrentIndex)

    def _updatePlotSelector(self, index):
        """Update the plot selector box"""
        # Repopulate the plot selector combobox with the stacks
        self.plotSelector.clear()
        for i in range(self.plotStack.count()):
            widget = self.plotStack.widget(i)
            name = widget.__class__.__name__

            self.plotSelector.addItem(name)

        # Set the combobox's active item to the currently active widget from
        # the plot stack
        current_index = self.plotStack.currentIndex()
        self.plotSelector.setCurrentIndex(current_index)

    def fileOpenDialog(self):
        """Open a file selection dialog.

        This slot is defined in the QT Designer.
        """
        # Get the filepath for the opened file
        kwargs = {'parent': self,
                  'caption': 'Open File',
                  'directory': str(Path.home()),
                  'filter': "NMRPipe Files (*.fid *.ft *.ft2 *.ft3)"}
        in_filepath, selected_filter = QFileDialog.getOpenFileName(**kwargs)

        if in_filepath:  # Must be a string with contents
            self.addSpectrum(in_filepath)

    def fileNotFoundDialog(self, filename):
        """Show a file not found dialog"""
        msg = QMessageBox()
        msg.setText(f"Could not find file '{filename}'")
        msg.setWindowTitle("File Not Found")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def addSpectrum(self, in_filepath, error_dialog=True):
        """Add a spectrum to the app.

        Parameters
        ----------
        in_filepath
            The filename for the spectrum to add/open.
        error_dialog
            Display an error dialog if the spectrum could not be added
        """
        # Create the spectrum and add it to the list of spectra
        spectrum = NMRPipeSpectrum(in_filepath)
        self.spectra.append(spectrum)

        # Create a stack view for the plot
        plot_widget = NMRSpectrumContour2DWidget(spectra=[spectrum])
        self.plotStack.addWidget(plot_widget)

