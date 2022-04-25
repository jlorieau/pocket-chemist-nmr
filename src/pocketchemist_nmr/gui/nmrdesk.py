"""
The root window for the NMRDesk application
"""
from pathlib import Path
import typing as t
from weakref import ReferenceType, ref

import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QMenuBar, QStatusBar,
                             QToolBar, QComboBox, QFileDialog, QMessageBox)
from PyQt6 import uic
from pyqtgraph import (PlotWidget, GraphicsLayoutWidget, IsocurveItem,
                       ImageItem, ViewBox)

from ..spectra import NMRSpectrum, NMRPipeSpectrum


class NMRSpectrumPlotWidget(GraphicsLayoutWidget):
    """A plot widget for an NMRSpectrum"""

    #: Viewpox for rendering the plots
    viewBox: ViewBox

    #: The spectrum to plot
    _spectrum: ReferenceType

    def __init__(self, spectrum: NMRSpectrum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectrum = spectrum
        data = spectrum.data.numpy()

        # Setup the widget
        self.viewBox = self.addViewBox()
        self.viewBox.setAspectLocked()
        img = ImageItem(data)
        self.viewBox.addItem(img)

        c = IsocurveItem(data=data, level=data.max() * 0.25,
                         pen='r')
        print(c)
        c.setParentItem(self.viewBox)
        self.viewBox.addItem(c)
        # c.setParentItem(img)
        # c.setZValue(10)

    @property
    def spectrum(self) -> NMRSpectrum:
        return self._spectrum()

    @spectrum.setter
    def spectrum(self, value: NMRSpectrum):
        self._spectrum = ref(value)


class NMRDeskWindow(QMainWindow):

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

    def __init__(self):
        super().__init__()

        # Setup mutable containers
        self.spectra = []

        # Load designer layout
        uic.loadUi(Path(__file__).parent / 'nmrdesk.ui', self)

        # Add a spectrum selector to the toolbar
        self.toolBar.addWidget(QComboBox())

        # Hide the status bar at the bottom of the window
        self.statusbar.setVisible(False)

        self.show()

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
        plot_widget = NMRSpectrumPlotWidget(spectrum=spectrum)
        self.plotStack.addWidget(plot_widget)

