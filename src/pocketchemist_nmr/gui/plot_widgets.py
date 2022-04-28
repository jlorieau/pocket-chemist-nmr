"""
Widgets for plotting
"""
import typing as t
from weakref import ReferenceType, ref

import numpy as np
from PyQt6.QtGui import QTransform, QFont, QPainterPath
from pyqtgraph import (PlotWidget, GraphicsLayout, PlotItem, IsocurveItem,
                       ImageItem, ViewBox, colormap)

from .funcs import isocurve
from .constants import MouseMode
from ..spectra import NMRSpectrum


class FasterIsocurveItem(IsocurveItem):
    """An IsocurveItem with a faster marching square implementation"""
    def generatePath(self):
        if self.data is None:
            self.path = None
            return

        if self.axisOrder == 'row-major':
            data = self.data.T
        else:
            data = self.data

        lines = isocurve(data, self.level, connected=True, extendToEdge=True)
        self.path = QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)


class FlexibleViewBox(ViewBox):
    """A view box with greater flexibility in its mouse and menu functions."""

    #: The current mouse navigation or selection mode
    mouseMode = MouseMode.NAVIGATION

    def showAxRect(self, ax, **kwargs):
        """The rectangle function called in 1-button mouse mode.

        This method highjacks the pyqtgraph's 1-button mouse mode so
        that it can be used for functionality like adding peaks.
        """
        if self.mouseMode.ADDPEAKS:
            pass
        else:
            # Conduct the zoom-in, as usual
            return super().showAxRect(ax, **kwargs)


class NMRSpectrumPlot(PlotWidget):
    """A generic widget base class for plotting NMR spectra"""

    #: Axis title font family
    axisTitleFontFamily = "Helvetica"

    #: Axis title font size (in pt)
    axisTitleFontSize = 16

    #: Axis label font
    axisLabelFontFamily = "Helvetica"

    #: Axis size of label fonts (in pt)
    axisLabelFontSize = 14

    #: The viewbox for the plot
    _viewBox: FlexibleViewBox

    #: The spectra to plot
    _spectra: t.List[ReferenceType[NMRSpectrum]]

    def __init__(self, spectra: t.List[NMRSpectrum], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wrap the spectra in a list if needed
        spectra = [spectra] if type(spectra) not in (list, tuple) else spectra

        # Setup the containers and data
        self.spectra = spectra

        # Setup an instance of the subclassed view box
        self._viewBox = FlexibleViewBox()

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

    def setMouseMode(self, mode: MouseMode):
        """Set the mouse mode"""
        self._viewBox.mouseMode = mode

        # Set the mouse mode for the view box
        if mode is MouseMode.ADDPEAKS:
            self._viewBox.setMouseMode(ViewBox.RectMode)
        else:
            self._viewBox.setMouseMode(ViewBox.PanMode)


class NMRSpectrumContour2D(NMRSpectrumPlot):
    """A plot widget for an NMRSpectrum"""

    #: Lock aspect ratio for the plot
    lockAspect: bool = True

    #: The number of contour levels to draw
    contourLevels = 10

    #: The type of contours to draw
    contourType = 'multiplicative'

    #: The increase factor for multiplicative contours
    contourFactor = 1.2

    #: The level for the first contours (positive and negative
    contourStartPositive = None
    contourStartNegative = None

    #: The scale of the maximum height in the data to use in populating
    #: contourStartPositive/contourStartNegative, if these are specified
    contourStartScale = 0.1

    #: Color maps to use for the positive/negative contours of the first,
    #: second, etc. spectra
    colormaps = (
        ('CET-L8', 'CET-L14'),  # blue->yellow,, black->green
        ('CET-L4', 'CET-L14'),  # red->yellow, black->green
        ('CET-L5', 'CET-L13'),  # green->white, black->red
        ('CET-R3', 'CET-L14'),  # blue->green>yellow->red,, black->green
        ('CET-L19', 'CET-L14'),  # white->red,, black->green
        ('CET-L6', 'CET-L13'),  # blue->white, black->red
    )

    #: The graphics layout for contours
    _layout: GraphicsLayout

    #: The plot item for contours
    _plotItem: PlotItem

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setup the graphics layout
        self._layout = GraphicsLayout()
        self.setCentralItem(self._layout)

        # Setup the plot item
        self._plotItem = PlotItem(viewBox=self._viewBox)
        self._layout.addItem(self._plotItem)

        # Load the contours
        self.loadContours()

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

    def getContourLevels(self) -> t.Tuple[t.Tuple[float, ...],
                                          t.Tuple[float, ...]]:
        """Calculate the contour levels

        Returns
        -------
        positive_contours, negative_contours
            The tuples for the data heights/intensities of the positive value
            and negative value contours.
        """
        positive_start = self.contourStartPositive
        negative_start = self.contourStartNegative

        # Calculate positive_start and negative_start values, if these aren't
        # specified
        if positive_start is None or negative_start is None:
            # Determine the maximum data height (intensity)
            max_height = 0.0
            for spectrum in self.spectra:
                data_max = float(max(abs(spectrum.data.max()),
                                     abs(spectrum.data.min())))
                max_height = data_max if data_max > max_height else max_height

            positive_start = max_height * self.contourStartScale
            negative_start = max_height * self.contourStartScale * -1.

            self.contourStartPositive = positive_start
            self.contourStartNegative = negative_start

        # Calculate contours according to the specified method
        if self.contourType == 'multiplicative':
            positive_contours = tuple(positive_start * self.contourFactor ** i
                                      for i in range(self.contourLevels))
            negative_contours = tuple(negative_start * self.contourFactor ** i
                                      for i in range(self.contourLevels))
            return positive_contours, negative_contours
        else:
            return tuple(), tuple()

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

        # Configure the axes
        labelFont = QFont(self.axisLabelFontFamily,
                          self.axisLabelFontSize)

        bottom = self._plotItem.getAxis('bottom')
        bottom.setLabel(self.xAxisTitle, 'ppm',
                        **{'font-family': self.axisTitleFontFamily,
                           'font-size': f'{self.axisLabelFontSize}pt'})
        bottom.setStyle(tickFont=labelFont)

        left = self._plotItem.getAxis('left')
        left.setLabel(text=self.yAxisTitle, units='ppm',
                      **{'font-family': self.axisTitleFontFamily,
                         'font-size': f'{self.axisLabelFontSize}pt'})
        left.setStyle(tickFont=labelFont)

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
        positive_contours, negative_contours = self.getContourLevels()
        cm_positive = colormap.get(self.colormaps[0][0])
        cm_negative = colormap.get(self.colormaps[0][1])

        for levels, cm in zip((positive_contours, negative_contours),
                              (cm_positive, cm_negative)):
            if len(levels) == 0:
                continue
            color_table = cm.getLookupTable(nPts=len(levels))

            for level, color in zip(levels, color_table):

                c = FasterIsocurveItem(data=data, level=level, pen=color)
                c.setParentItem(img)
                c.generatePath()
