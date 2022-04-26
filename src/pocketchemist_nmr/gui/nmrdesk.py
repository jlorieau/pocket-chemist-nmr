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
from PyQt6.QtGui import QTransform
from pyqtgraph import (PlotWidget, GraphicsLayout, PlotItem, IsocurveItem,
                       ImageItem, ViewBox)

from ..spectra import NMRSpectrum, NMRPipeSpectrum


class NMRSpectrumContour2DWidget(PlotWidget):
    """A plot widget for an NMRSpectrum"""

    #: Lock aspect ratio for the plot
    lockAspect: bool = True

    #: Aspect ratio for the plot
    aspect: float = 0.1 * (3. / 2.)

    #: The spectra to plot
    _spectra: t.List[ReferenceType]

    #: The graphics layout for contours
    _layout: GraphicsLayout

    #: The plot item for contours
    _plotItem: PlotItem

    def __init__(self, spectra: t.List[NMRSpectrum], *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wrap the spectra in a list if needed
        spectra = [spectra] if type(spectra) not in (list, tuple) else spectra

        # Setup the containers and data
        self._curves = []
        self.spectra = spectra

        # Setup the graphics layout
        self._layout = GraphicsLayout()
        self.setCentralItem(self._layout)

        # Setup the plot item
        self._plotItem = PlotItem()
        self._layout.addItem(self._plotItem)

        # Load the contours
        self.loadContours()

    @property
    def spectra(self) -> t.List[NMRSpectrum]:
        spectra = [spectrum() for spectrum in self._spectra]
        return [spectrum for spectrum in spectra if spectrum is not None]

    @spectra.setter
    def spectra(self, value: t.List[NMRSpectrum]):
        # Initialize container, if needed
        if getattr(self, '_spectra', None) is None:
            self._spectra = []

        self._spectra.clear()
        self._spectra += [ref(spectrum) for spectrum in value]

    @property
    def xAxisTitle(self):
        """The label for the x-axis"""
        spectra = self.spectra
        return spectra[0].label[0] if spectra is not None else ''

    @property
    def yAxisTitle(self):
        """The label for the x-axis"""
        spectra = self.spectra
        return spectra[0].label[1] if spectra is not None else ''

    def loadContours(self):
        """Load the contour levels for the spectrum"""
        # Retrieve the spectrum from the weakref
        spectrum = self.spectra[0]
        if spectrum is None:
            return None

        # Retrieve the data to plot contours. The axes need to be inverted
        # for axes in ppm and Hz, so the data must be flipped too.
        data = spectrum.data.numpy()
        data = np.flipud(np.fliplr(data))  # Flip x- and y-axes

        # Retrieve the x-axis and y-axis ranges
        x_min, x_max, = spectrum.range_ppm[0]
        y_min, y_max, = spectrum.range_ppm[1]
        x_range = abs(x_min - x_max)  # spectral width
        y_range = abs(y_min - y_max)  # spectral width

        # Reset the plotItem
        self._plotItem.clear()

        # Setup the plot and axis displays
        self._plotItem.vb.setAspectLocked(lock=self.lockAspect,
                                          ratio=y_range / x_range)
        self._plotItem.setLabel(axis='bottom', text=self.xAxisTitle)
        self._plotItem.setLabel(axis='left', text=self.yAxisTitle)

        # Setup the axes for the plot item
        self._plotItem.setXRange(x_min, x_max)
        self._plotItem.setYRange(y_min, y_max)

        # Flip the axes, needed for ppm and Hz data in NMR data
        self._plotItem.invertX(True)
        self._plotItem.invertY(True)

        # Load the data as an image and scale/translate from the index units
        # of the data to the units of the final spectrum (ppm or Hz)
        img = ImageItem()
        tr = QTransform()
        tr.scale(x_range / data.shape[0], y_range / data.shape[1])
        tr.translate(x_max * data.shape[0] / x_range,
                     y_max * data.shape[1] / y_range)
        img.setTransform(tr)
        self._plotItem.addItem(img)

        # Add the contours to the plot item
        c = IsocurveItem(data=data, level=data.max() * 0.10, pen='r')
        c.setParentItem(img)


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
        plot_widget = NMRSpectrumContour2DWidget(spectra=[spectrum])
        self.plotStack.addWidget(plot_widget)

