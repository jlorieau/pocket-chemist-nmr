"""
Widgets for plotting
"""
import typing as t
from weakref import ReferenceType, ref

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QTransform, QFont, QPainterPath, QPen
from PyQt6.QtWidgets import QLineEdit
from pyqtgraph import (PlotWidget, GraphicsLayout, PlotItem, mkPen,
                       IsocurveItem, ImageItem, InfiniteLine, ViewBox, colormap)

from .funcs import isocurve
from .constants import MouseInteraction
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

        lines = isocurve(data.real, self.level, connected=True,
                         extendToEdge=True)
        self.path = QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)


class FlexibleViewBox(ViewBox):
    """A view box with greater flexibility in its mouse and menu functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state['mouseInteraction'] = MouseInteraction.NAVIGATION

    def showAxRect(self, ax, **kwargs):
        """The rectangle function called in 1-button mouse mode.

        This method highjacks the pyqtgraph's 1-button mouse mode so
        that it can be used for functionality like adding peaks.
        """
        if self.getMouseInteraction() is MouseInteraction.ADDPEAKS:
            pass
        else:
            # Conduct the zoom-in, as usual
            return super().showAxRect(ax, **kwargs)

    def setMouseInteraction(self, mode):
        self.state['mouseInteraction'] = mode

    def getMouseInteraction(self):
        """Get the current mouse mode"""
        return self.state.get('mouseInteraction', MouseInteraction.NAVIGATION)


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

    #: The statusbar lineedit widget
    statusbar: QLineEdit

    #: The viewbox for the plot
    viewBox: FlexibleViewBox

    #: The spectra to plot
    _spectra: t.List[ReferenceType[NMRSpectrum]]

    def __init__(self, spectra: t.List[NMRSpectrum], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wrap the spectra in a list if needed
        spectra = [spectra] if type(spectra) not in (list, tuple) else spectra

        # Setup the containers and data
        self.spectra = spectra

        # Setup an instance of the subclassed view box
        self.viewBox = FlexibleViewBox()

        # Setup the statusbar line edit widget
        self.statusbar = QLineEdit()

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

    def setMouseInteraction(self, mode: MouseInteraction):
        """Set the mouse interaction mode"""
        self.viewBox.setMouseInteraction(mode)

        # Set the mouse mode for the view box
        if mode is MouseInteraction.ADDPEAKS:
            self.viewBox.setMouseMode(ViewBox.RectMode)
        else:
            self.viewBox.setMouseMode(ViewBox.PanMode)

    def getMouseInteraction(self):
        """Get the current mouse interaction mode"""
        return self.viewBox.getMouseInteraction()


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

    #: The pen for drawing the crosshair
    crosshairPen: QPen = mkPen(color="#666666")

    #: The pen for drawing htrace and vtrace lines
    tracePen: QPen = mkPen(color="aaaaaa")

    #: The graphics layout for contours
    _layout: GraphicsLayout

    #: The plot item for contours
    _plotItem: PlotItem

    #: The crosshair items
    _crosshair: t.Optional[t.List[InfiniteLine]]

    #: The htrace/vtrace selector line
    _selectorLine = t.Optional[InfiniteLine]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make copies of mutables
        self.crosshairPen = QPen(self.crosshairPen)
        self.tracePen = QPen(self.tracePen)

        # Setup the graphics layout
        self._layout = GraphicsLayout()
        self.setCentralItem(self._layout)

        # Setup the plot item
        self._plotItem = PlotItem(viewBox=self.viewBox)
        self._layout.addItem(self._plotItem)

        # Configure the axes
        self._setupAxes()

        # Load the contours
        self._loadContours()

        # Add the crosshair
        self._addCrosshair()

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

    def setMouseInteraction(self, mode: MouseInteraction):
        """Set the mouse mode"""
        super().setMouseInteraction(mode)

        # Reset vtrace/htrace modes
        selectorLine = getattr(self, '_selectorLine', None)
        if selectorLine is not None:
            self._plotItem.removeItem(selectorLine)
            self._selectorLine = None

        if mode is MouseInteraction.HTRACE:
            # Add a horizontal line for HTRACE mode
            self._selectorLine = InfiniteLine(angle=0, movable=False,
                                              pen=self.tracePen)
            self._plotItem.addItem(self._selectorLine)
        elif mode is MouseInteraction.VTRACE:
            # Add a vertical line for VTRACE mode
            self._selectorLine = InfiniteLine(angle=90, movable=False,
                                              pen=self.tracePen)
            self._plotItem.addItem(self._selectorLine)

    def mousePressEvent(self, ev):
        """Changes for mouse clicks"""
        pos = QPointF(ev.pos())

        # Do nothing if it's not within the bounding box
        if not self._plotItem.sceneBoundingRect().contains(pos):
            return super().mousePressEvent(ev)

        # Convert from pixels to the plot's coordinates
        mousePoint = self.viewBox.mapSceneToView(pos)

        # Update the HTRACE/VTRACE selector line, if available
        selectorLine = getattr(self, '_selectorLine', None)
        currentMouseInteraction = self.getMouseInteraction()

        if (currentMouseInteraction is MouseInteraction.HTRACE and
           selectorLine is not None):
            selectorLine.setPos(mousePoint.y())

        elif (currentMouseInteraction is MouseInteraction.VTRACE and
              selectorLine is not None):
            selectorLine.setPos(mousePoint.x())

        # Continue as normal
        return super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Changes for mouse moves"""
        pos = QPointF(ev.pos())

        # Do nothing if it's not within the bounding box
        if not self._plotItem.sceneBoundingRect().contains(pos):
            return super().mouseMoveEvent(ev)

        mousePoint = self.viewBox.mapSceneToView(pos)
        crosshair = getattr(self, '_crosshair', None)

        # Adjust the crosshair position
        if crosshair is not None:
            # Switch to the coordinates of the view box
            crosshair[0].setPos(mousePoint.y())  # hLine
            crosshair[1].setPos(mousePoint.x())  # vLine

        # Update the statusbar
        self.statusbar.setText(f"({mousePoint.x():.3f}, "
                               f"{mousePoint.y():.3f})")

        return super().mouseMoveEvent(ev)

    def _setupAxes(self):
        """Configure the axes"""
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

        # Flip the axes, needed for ppm and Hz data in NMR data
        self._plotItem.invertX(True)
        self._plotItem.invertY(True)

    def _addCrosshair(self):
        """Add the crosshair to the plot"""
        crosshair = getattr(self, '_crosshair', None)

        if crosshair is None:
            # Setup the crosshair
            hLine = InfiniteLine(angle=0, movable=False, pen=self.crosshairPen)
            vLine = InfiniteLine(angle=90, movable=False, pen=self.crosshairPen)
            self._plotItem.addItem(hLine)
            self._plotItem.addItem(vLine)
            self._crosshair = [hLine, vLine]

    def _removeCrosshair(self):
        """Remove the crosshair from the plot"""
        crosshair = getattr(self, '_crosshair', None)

        if crosshair is not None:
            hLine, vLine = crosshair
            self._plotItem.removeItem(hLine)
            self._plotItem.removeItem(vLine)
            self._crosshair = None

    def _getContourLevels(self) -> t.Tuple[t.Tuple[float, ...],
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
                data_max = float(max(abs(spectrum.data.real.max()),
                                     abs(spectrum.data.real.min())))
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

    def _loadContours(self):
        """Load the contour levels for the spectrum"""
        # Retrieve the spectrum from the weakref
        spectra = self.spectra
        if len(spectra) == 0:
            return None

        # Reset the plotItem
        self._plotItem.clear()

        # Setup the ranges for the plots. The x_ranges and y_ranges contain
        # 3-ples with (min, max, width) values
        x_ranges, y_ranges = [], []
        for spectrum in spectra:
            x_min, x_max = spectrum.range_ppm[0]
            y_min, y_max = spectrum.range_ppm[1]
            x_ranges.append((x_min, x_max, abs(x_min - x_max)))
            y_ranges.append((y_min, y_max, abs(y_min - y_max)))

        # Find the smallest and largest x_min/x_max/y_min/y_max to encompass
        # the range for all spectra
        x_min = min(x[0] for x in x_ranges)
        x_max = max(x[1] for x in x_ranges)
        y_min = min(y[0] for y in y_ranges)
        y_max = max(y[1] for y in y_ranges)

        # Setup the axes for the plot item
        self._plotItem.setXRange(x_min, x_max)
        self._plotItem.setYRange(y_min, y_max)

        # Set the aspect ratio for the plot view based on the largest
        # x_range and y_range
        max_x_width = max(x[2] for x in x_ranges)
        max_y_width = max(y[2] for y in y_ranges)
        self._plotItem.vb.setAspectLocked(lock=self.lockAspect,
                                          ratio=max_y_width / max_x_width)

        # Create the contours for each spectrum
        for i, (spectrum, x_range, y_range), in enumerate(zip(spectra, x_ranges,
                                                              y_ranges)):
            data = spectrum.data.numpy()  # view as numpy data
            data = np.flipud(np.fliplr(data))  # Flip x- and y-axes

            # Get the widths of the axes
            x_min, y_min = x_range[0], y_range[0]
            x_max, y_max = x_range[1], y_range[1]
            x_width, y_width = x_range[2], y_range[2]

            # Load the data as an image and scale/translate from the index units
            # of the data to the units of the final spectrum (ppm or Hz)
            img = ImageItem()
            tr = QTransform()
            tr.scale(x_width / data.shape[0], y_width / data.shape[1])
            tr.translate(x_max * data.shape[0] / x_width,
                         y_max * data.shape[1] / y_width)
            img.setTransform(tr)
            self._plotItem.addItem(img)

            # Retrieve the next contour map, or return a default
            cm = (self.colormaps[i] if i < len(self.colormaps) else
                  self.colormaps[-1])
            positive_contours, negative_contours = self._getContourLevels()
            cm_positive = colormap.get(cm[0])
            cm_negative = colormap.get(cm[1])

            # Add the contours to the plot item
            for levels, cm in zip((positive_contours, negative_contours),
                                  (cm_positive, cm_negative)):
                if len(levels) == 0:
                    continue
                color_table = cm.getLookupTable(nPts=len(levels))

                for level, color in zip(levels, color_table):

                    c = FasterIsocurveItem(data=data, level=level, pen=color)
                    c.setParentItem(img)
                    c.generatePath()
