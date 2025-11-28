# qt_viewport.py
from qtpy.QtWidgets import QWidget, QVBoxLayout
from napari import Viewer
from qt_viewport_buttons import QtViewportButtons # Your custom buttons

class QtViewport(QWidget):
    def __init__(self, viewer: Viewer, title: str, parent=None, leave_controls=False):
        super().__init__(parent)
        self.viewer = viewer

        # 1. Extract the native Qt widget from the napari viewer
        # We treat the napari viewer as a component provider
        qt_viewer = viewer.window._qt_viewer

        # 2. Custom Buttons attached to this viewer
        self.buttons = QtViewportButtons(viewer)

        # 3. Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)

        # Custom Title Bar / Buttons
        layout.addWidget(self.buttons)

        # The actual Napari Canvas + Dims slider
        layout.addWidget(qt_viewer)

        self.setLayout(layout)

        # Hide the specific dock widgets for this viewer instance
        # so they don't float around. We only want the canvas.
        qt_viewer.dockLayerList.setVisible(leave_controls)
        qt_viewer.dockLayerControls.setVisible(leave_controls)
        qt_viewer.dockConsole.setVisible(leave_controls)