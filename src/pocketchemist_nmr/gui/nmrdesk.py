"""
The root window for the NMRDesk application
"""
from pathlib import Path
import typing as t

from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QMenuBar, QStatusBar,
                             QToolBar, QComboBox, QFileDialog, QMessageBox,
                             QWidget, QSizePolicy, QMenu)
from PyQt6.QtGui import QAction, QActionGroup, QIcon
from PyQt6 import uic

from .plot_widgets import NMRSpectrumPlot, NMRSpectrumContour2D
from .constants import MouseMode
from ..spectra import NMRSpectrum, NMRPipeSpectrum


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

    #: The selector for the plot displayed
    plotSelector: QComboBox

    #: The width (in number of characters) of the plot selector combo box
    plotSelectorWidth = 50

    #: The current mouse mouse mode
    currentMouseMode: MouseMode = MouseMode.NAVIGATION

    #: Paths for icons
    icon_paths = (Path(__file__).parent / 'assets' / 'icons8',)

    #: A dict with icons
    _icons: t.Dict[str, QIcon]

    #: The action group to specify the current mouse mose
    _mouseMode: QActionGroup

    def __init__(self):
        super().__init__()

        # Setup mutable containers
        self.spectra = []
        self._icons = dict()

        # Load icons and the designer layout and setup the window
        uic.loadUi(Path(__file__).parent / 'nmrdesk.ui', self)
        self._loadIcons()

        self._setupToolbar()
        self.statusbar.setVisible(False)  # Hide status bar

        self.show()

    def _loadIcons(self):
        """Load icons"""
        # Setup the container
        if getattr(self, '_icons', None) is None:
            self._icons = dict()

        # Load icons into the container
        for icon_path in self.icon_paths:
            for filepath in icon_path.glob('*.png'):
                name = filepath.name.replace('icons8-', '').replace('.png', '')
                self._icons[name] = QIcon(str(filepath))

    def _setupToolbar(self):
        """Setup the toolbar with extra widgets not added by designer"""
        # Set the mouse mode actions
        self._setupMouseModeActionGroup()

        # Add a horizontal spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Minimum)
        self.toolBar.addWidget(spacer)

        # Add plot selector combobox
        self._setupToolbarPlotSelector()

    def _setupToolbarPlotSelector(self):
        """Setup the PlotSelector combobox for the toolbar"""
        self.plotSelector = QComboBox()

        # Set the width of the widget
        metrics = self.plotSelector.fontMetrics()
        width = metrics.boundingRect(' ' * self.plotSelectorWidth).width()
        self.plotSelector.setMinimumWidth(width)

        self.toolBar.addWidget(self.plotSelector)  # add to toolbar

        # Changes to the self.plotStack should be reflected in the combobox
        self.plotStack.currentChanged.connect(self._updatePlotSelector)
        self.plotSelector.activated.connect(self.plotStack.setCurrentIndex)

    def _setupMouseModeActionGroup(self):
        """Setup the mouse mode action group"""
        # Create actions for the modes
        navigate = QAction(self._icons['navigate-40'],
                           MouseMode.NAVIGATION.value, self)
        addpeaks = QAction(self._icons['add-40'],
                           MouseMode.ADDPEAKS.value, self)
        actions = (navigate, addpeaks)

        # Set the actions to be checkable
        navigate.setChecked(True)
        for action in actions:
            action.setCheckable(True)

        # Group these into an ActionGroup
        self._mouseMode = QActionGroup(self)
        for action in actions:
            self._mouseMode.addAction(action)

        # Connect the signal. The signal sends a QAction, which needs to be
        # converted to a MouseMode
        enums = {enum.value: enum for enum in MouseMode}
        self._mouseMode.triggered.connect(lambda a:
                                          self.setMouseMode(enums[a.text()]))

        # Add the group to the toolbar
        self.toolBar.addActions(actions)

        # Add the group to the edit menu
        edit = self.menubar.findChild(QMenu, "menuEdit")
        mouseMode = QMenu("Mouse Mode")
        mouseMode.addActions(actions)
        edit.addMenu(mouseMode)

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

    def setMouseMode(self, mode: MouseMode):
        """Set the current mouse mode"""
        self.currentMouseMode = mode

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
        plot_widget = NMRSpectrumContour2D(spectra=[spectrum])
        self.plotStack.addWidget(plot_widget)

        # Connect the plot's signals
        enums = {enum.value: enum for enum in MouseMode}
        slot = lambda a: plot_widget.setMouseMode(enums[a.text()])
        self._mouseMode.triggered.connect(slot)
