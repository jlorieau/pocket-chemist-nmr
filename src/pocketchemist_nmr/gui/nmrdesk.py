"""
The root window for the NMRDesk application
"""
from PyQt6.QtWidgets import (QWidget, QMainWindow, QPushButton, QGridLayout,
                             QLineEdit,QListWidget)
from pyqtgraph import PlotWidget


class NMRDeskWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NMRDesk")

        # Create layout
        layout = QGridLayout()

        # Create widgets
        plot = PlotWidget()

        # Add widgets to the layout in their proper positions
        layout.addWidget(plot, 0, 0)  # plot on right side spanning 3 rows

        # Set the central widget of the Window.
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
