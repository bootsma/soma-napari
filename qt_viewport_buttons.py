import sys
from typing import TYPE_CHECKING
from functools import partial

import napari.components.viewer_model
from qtpy.QtWidgets import (
    QWidget,
    QFrame,
    QPushButton,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
)
from qtpy.QtCore import Qt

# --- Imports from napari ---
# We need to recreate the QtViewportButtons class here
# as the environment is stateless.
try:
    from napari.components.viewer_model import ViewerModel
    from napari._qt.widgets.qt_dims import QtDims
    from napari._vispy import VispyCanvas, create_vispy_layer
    from napari.utils.translations import trans
    from napari.utils.events import Event

    # Imports for QtViewportButtons and its popups
    from napari._qt.widgets.qt_viewer_buttons import (
        QtViewerPushButton,
        labeled_double_slider,
        enum_combobox,
        help_tooltip,
    )
    # Add the KeymapHandler import
    from napari.utils.key_bindings import KeymapHandler

    from napari._qt.dialogs.qt_modal import QtPopup
    from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
    from napari._qt.widgets.qt_spinbox import QtSpinBox
    from napari._qt.widgets.qt_tooltip import QtToolTipLabel
    from napari.utils.camera_orientations import (
        DepthAxisOrientation,
        DepthAxisOrientationStr,
        HorizontalAxisOrientation,
        HorizontalAxisOrientationStr,
        VerticalAxisOrientation,
        VerticalAxisOrientationStr,
    )
except ImportError:
    print(
        "Could not import napari components. "
        "Please ensure napari is installed."
    )
    sys.exit(1)



class QtViewportButtons(QFrame):
    """Button controls for a napari viewport.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer model containing the rendered scene, layers, and controls
        for this specific viewport.
    """

    def __init__(self, viewer: napari.components.viewer_model.ViewerModel) -> None:
        super().__init__()

        self.viewer = viewer



        rdb = QtViewerPushButton(
            'roll',
            tooltip=trans._('Roll dimensions'),
            slot=self._on_roll,
        )
        self.rollDimsButton = rdb
        rdb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        rdb.customContextMenuRequested.connect(self._open_roll_popup)

        self.transposeDimsButton = QtViewerPushButton(
            'transpose',
            tooltip=trans._('Transpose dimensions'),
            slot=self.viewer.dims.transpose,
        )

        self.resetViewButton = QtViewerPushButton(
            'home',
            tooltip=trans._('Reset view'),
            slot=self.viewer.reset_view,
        )

        gvb = QtViewerPushButton(
            'grid_view_button',
            tooltip=trans._('Toggle grid mode'),
            slot=self._toggle_grid,
        )
        self.gridViewButton = gvb
        gvb.setCheckable(True)
        gvb.setChecked(viewer.grid.enabled)
        gvb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        gvb.customContextMenuRequested.connect(self._open_grid_popup)

        @self.viewer.grid.events.enabled.connect
        def _set_grid_mode_checkstate(event):
            gvb.setChecked(event.value)

        ndb = QtViewerPushButton(
            'ndisplay_button',
            tooltip=trans._('Toggle 2D/3D display'),
            slot=self._toggle_ndisplay,
        )
        self.ndisplayButton = ndb

        ndb.setCheckable(True)
        ndb.setChecked(self.viewer.dims.ndisplay == 3)
        ndb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        ndb.customContextMenuRequested.connect(self.open_ndisplay_camera_popup)

        @self.viewer.dims.events.ndisplay.connect
        def _set_ndisplay_mode_checkstate(event):
            ndb.setChecked(event.value == 3)


        adb = QtViewerPushButton(
            'coordinate_axes',
            tooltip=trans._('Toggle axes and scale bar display'),
            slot=self._toggle_display_axes,
        )
        self.axesDisplayButton = adb
        adb.setCheckable(True)
        adb.setChecked(viewer.axes.visible)
        #db.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        #adb.customContextMenuRequested.connect(self._display_axes)

        @self.viewer._overlays.events.added.connect
        def _a(event):
            print('a',event)


        @self.viewer._overlays.events.removed.connect
        def _B(event):
            print('b',event)

        self.view_type = QLabel('AXIAL')
        self.view_type.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._set_view_type()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ndisplayButton)
        layout.addWidget(self.rollDimsButton)
        layout.addWidget(self.transposeDimsButton)
        layout.addWidget(self.gridViewButton)
        layout.addWidget(self.resetViewButton)
        layout.addWidget(self.axesDisplayButton)
        layout.addWidget(self.view_type)
        layout.addStretch(0)
        self.setLayout(layout)
        #self.setFixedHeight(self.rollDimsButton.sizeHint().height())

    def _toggle_display_axes(self):

        self.viewer.scale_bar.visible = not self.viewer.scale_bar.visible

        self.viewer.axes.visible = not self.viewer.axes.visible

    def _on_roll(self):
        self.viewer.dims.roll()
        self._set_view_type()


    def _set_view_type(self):
        if len(self.viewer.dims.order) == 3:
            if self.viewer.dims.order[0] == 0:
                self.view_type.setText('AXIAL')
            elif self.viewer.dims.order[0] == 1:
                self.view_type.setText('CORONAL')  # Assuming the second dimension is coronal
            elif self.viewer.dims.order[0] == 2:
                self.view_type.setText('SAGITTAL')  # Assuming the third dimension is sagittal
            else:
                self.view_type.setText('UNSPECIFIED')
        else:
            self.view_type.setText(str(self.viewer.dims.order))
    def _toggle_grid(self):
        """Toggle grid mode on this viewport's viewer model."""
        self.viewer.grid.enabled = not self.viewer.grid.enabled

    def _toggle_ndisplay(self):
        """Toggle ndisplay mode on this viewport's viewer model."""
        self.viewer.dims.ndisplay = (
            3 if self.viewer.dims.ndisplay == 2 else 2
        )

    # --- All popup methods copied directly from QtViewerButtons ---
    #

    def _position_popup_inside_viewer(
            self, popup: QtPopup, button: QPushButton
    ) -> None:
        button_rect = button.rect()
        button_pos = button.mapToGlobal(button_rect.topLeft())
        popup.move_to(
            (
                button_pos.x(),
                button_pos.y() - popup.sizeHint().height() - 5,
                popup.sizeHint().width(),
                popup.sizeHint().height(),
            )
        )
        popup.show()

    def _add_3d_camera_controls(
            self,
            popup: QtPopup,
            grid_layout: QGridLayout,
    ) -> None:
        self.perspective = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.perspective,
            value_range=(0, 90),
            callback=self._update_perspective,
        )
        perspective_help_symbol = help_tooltip(
            parent=popup,
            text='Controls perspective projection strength. 0 is orthographic, larger values increase perspective effect.',
        )
        self.rx = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.angles[0],
            value_range=(-180, 180),
            callback=partial(self._update_camera_angles, 0),
        )
        self.ry = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.angles[1],
            value_range=(-89, 89),
            callback=partial(self._update_camera_angles, 1),
        )
        self.rz = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.angles[2],
            value_range=(-180, 180),
            callback=partial(self._update_camera_angles, 2),
        )
        angle_help_symbol = help_tooltip(
            parent=popup,
            text='Controls the rotation angles around each axis in degrees.',
        )
        grid_layout.addWidget(QLabel(trans._('Perspective:')), 2, 0)
        grid_layout.addWidget(self.perspective, 2, 1)
        grid_layout.addWidget(perspective_help_symbol, 2, 2)
        grid_layout.addWidget(QLabel(trans._('Angles    X:')), 3, 0)
        grid_layout.addWidget(self.rx, 3, 1)
        grid_layout.addWidget(angle_help_symbol, 3, 2)
        grid_layout.addWidget(QLabel(trans._('             Y:')), 4, 0)
        grid_layout.addWidget(self.ry, 4, 1)
        grid_layout.addWidget(QLabel(trans._('             Z:')), 5, 0)
        grid_layout.addWidget(self.rz, 5, 1)

    def _add_shared_camera_controls(
            self,
            popup: QtPopup,
            grid_layout: QGridLayout,
    ) -> None:
        self.zoom = labeled_double_slider(
            parent=popup,
            value=self.viewer.camera.zoom,
            value_range=(0.01, 100),
            decimals=2,
            callback=self._update_zoom,
        )
        zoom_help_symbol = help_tooltip(
            parent=popup,
            text='Controls zoom level of the camera. Larger values zoom in, smaller values zoom out.',
        )
        grid_layout.addWidget(QLabel(trans._('Zoom:')), 1, 0)
        grid_layout.addWidget(self.zoom, 1, 1)
        grid_layout.addWidget(zoom_help_symbol, 1, 2)

    def _add_orientation_controls(
            self,
            popup: QtPopup,
            grid_layout: QGridLayout,
    ) -> None:
        orientation_widget = QWidget(popup)
        orientation_layout = QHBoxLayout()
        orientation_layout.setContentsMargins(0, 0, 0, 0)

        self.vertical_combo = enum_combobox(
            parent=popup,
            enum_class=VerticalAxisOrientation,
            current_enum=self.viewer.camera.orientation[1],
            callback=partial(
                self._update_orientation, VerticalAxisOrientation
            ),
        )
        self.horizontal_combo = enum_combobox(
            parent=popup,
            enum_class=HorizontalAxisOrientation,
            current_enum=self.viewer.camera.orientation[2],
            callback=partial(
                self._update_orientation, HorizontalAxisOrientation
            ),
        )

        if self.viewer.dims.ndisplay == 2:
            orientation_layout.addWidget(self.vertical_combo)
            orientation_layout.addWidget(self.horizontal_combo)
            self.orientation_help_symbol = help_tooltip(
                parent=popup,
                text='Controls the orientation of the vertical and horizontal camera axes.',
            )
        else:
            self.depth_combo = enum_combobox(
                parent=popup,
                enum_class=DepthAxisOrientation,
                current_enum=self.viewer.camera.orientation[0],
                callback=partial(
                    self._update_orientation, DepthAxisOrientation
                ),
            )
            orientation_layout.addWidget(self.depth_combo)
            orientation_layout.addWidget(self.vertical_combo)
            orientation_layout.addWidget(self.horizontal_combo)
            self.orientation_help_symbol = help_tooltip(
                parent=popup,
                text='',  # updated dynamically
            )
            self._update_handedness_help_symbol()
            self.viewer.camera.events.orientation.connect(
                self._update_handedness_help_symbol
            )

        orientation_widget.setLayout(orientation_layout)
        grid_layout.addWidget(QLabel(trans._('Orientation:')), 0, 0)
        grid_layout.addWidget(orientation_widget, 0, 1)
        grid_layout.addWidget(self.orientation_help_symbol, 0, 2)

    def _update_handedness_help_symbol(self, event=None) -> None:
        handedness = self.viewer.camera.handedness
        tooltip_text = (
            'Controls the orientation of the depth, vertical, and horizontal camera axes.\n'
            'Default is right-handed (towards, down, right).\n'
            'Default prior to 0.6.0 was left-handed (away, down, right).\n'
            f'Currently orientation is {handedness.value}-handed.'
        )
        self.orientation_help_symbol.setToolTip(tooltip_text)
        self.orientation_help_symbol.setObjectName(
            f'{handedness.value}hand_label'
        )
        self.orientation_help_symbol.style().polish(
            self.orientation_help_symbol
        )

    def open_ndisplay_camera_popup(self) -> None:
        popup = QtPopup(self)
        grid_layout = QGridLayout()
        self._add_orientation_controls(popup, grid_layout)
        self._add_shared_camera_controls(popup, grid_layout)
        if self.viewer.dims.ndisplay == 3:
            self._add_3d_camera_controls(popup, grid_layout)
        popup.frame.setLayout(grid_layout)
        self._position_popup_inside_viewer(popup, self.ndisplayButton)

    def _update_orientation(
            self,
            orientation_type: type[
                DepthAxisOrientation
                | VerticalAxisOrientation
                | HorizontalAxisOrientation
                ],
            orientation_value: (
                    DepthAxisOrientationStr
                    | VerticalAxisOrientationStr
                    | HorizontalAxisOrientationStr
            ),
    ) -> None:
        axes = (
            DepthAxisOrientation,
            VerticalAxisOrientation,
            HorizontalAxisOrientation,
        )
        axis_to_update = axes.index(orientation_type)
        new_orientation = list(self.viewer.camera.orientation)
        new_orientation[axis_to_update] = orientation_type(orientation_value)
        self.viewer.camera.orientation = tuple(new_orientation)

    def _update_camera_angles(self, idx: int, value: float) -> None:
        angles = list(self.viewer.camera.angles)
        angles[idx] = value
        self.viewer.camera.angles = tuple(angles)

    def _update_zoom(self, value: float) -> None:
        self.viewer.camera.zoom = value

    def _update_perspective(self, value: float) -> None:
        self.viewer.camera.perspective = value

    def _open_roll_popup(self):
        pop = QtPopup(self)
        dim_sorter = QtDimsSorter(self.viewer.dims, pop)
        dim_sorter.setObjectName('dim_sorter')
        layout = QHBoxLayout()
        layout.addWidget(dim_sorter)
        pop.frame.setLayout(layout)
        pop.show_above_mouse()

    def _open_grid_popup(self):
        popup = QtPopup(self)
        grid_stride = QtSpinBox(popup)
        grid_width = QtSpinBox(popup)
        grid_height = QtSpinBox(popup)
        grid_spacing = QDoubleSpinBox(popup)
        shape_help_symbol = QtToolTipLabel(self)
        stride_help_symbol = QtToolTipLabel(self)
        spacing_help_symbol = QtToolTipLabel(self)

        shape_help_msg = trans._(
            'Number of rows and columns in the grid.\n'
            'A value of -1 for either or both of width and height will trigger an\n'
            'auto calculation of the necessary grid shape to appropriately fill\n'
            'all the layers at the appropriate stride. 0 is not a valid entry.'
        )
        stride_help_msg = trans._(
            'Number of layers to place in each grid viewbox before moving on to the next viewbox.\n'
            'A negative stride will cause the order in which the layers are placed in the grid to be reversed.\n'
            '0 is not a valid entry.'
        )
        spacing_help_msg = trans._(
            'The amount of spacing between grid viewboxes.\n'
            'If between 0 and 1, it is interpreted as a proportion of the size of the viewboxes.\n'
            'If equal or greater than 1, it is interpreted as screen pixels.'
        )

        # ... (rest of popup code is identical and self-contained) ...
        #

        stride_min = self.viewer.grid.__fields__['stride'].type_.ge
        stride_max = self.viewer.grid.__fields__['stride'].type_.le
        stride_not = self.viewer.grid.__fields__['stride'].type_.ne
        grid_stride.setObjectName('gridStrideBox')
        grid_stride.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_stride.setRange(stride_min, stride_max)
        grid_stride.setProhibitValue(stride_not)
        grid_stride.setValue(self.viewer.grid.stride)
        grid_stride.valueChanged.connect(self._update_grid_stride)
        self.grid_stride_box = grid_stride

        width_min = self.viewer.grid.__fields__['shape'].sub_fields[1].type_.ge
        width_not = self.viewer.grid.__fields__['shape'].sub_fields[1].type_.ne
        grid_width.setObjectName('gridWidthBox')
        grid_width.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_width.setMinimum(width_min)
        grid_width.setProhibitValue(width_not)
        grid_width.setValue(self.viewer.grid.shape[1])
        grid_width.valueChanged.connect(self._update_grid_width)
        self.grid_width_box = grid_width

        height_min = (
            self.viewer.grid.__fields__['shape'].sub_fields[0].type_.ge
        )
        height_not = (
            self.viewer.grid.__fields__['shape'].sub_fields[0].type_.ne
        )
        grid_height.setObjectName('gridStrideBox')
        grid_height.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_height.setMinimum(height_min)
        grid_height.setProhibitValue(height_not)
        grid_height.setValue(self.viewer.grid.shape[0])
        grid_height.valueChanged.connect(self._update_grid_height)
        self.grid_height_box = grid_height

        spacing_min = self.viewer.grid.__fields__['spacing'].type_.ge
        spacing_max = self.viewer.grid.__fields__['spacing'].type_.le
        spacing_step = self.viewer.grid.__fields__['spacing'].type_.step
        grid_spacing.setObjectName('gridSpacingBox')
        grid_spacing.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_spacing.setMinimum(spacing_min)
        grid_spacing.setMaximum(spacing_max)
        grid_spacing.setValue(self.viewer.grid.spacing)
        grid_spacing.setDecimals(2)
        grid_spacing.setSingleStep(spacing_step)
        grid_spacing.valueChanged.connect(self._update_grid_spacing)
        self.grid_spacing_box = grid_spacing

        shape_help_symbol.setObjectName('help_label')
        shape_help_symbol.setToolTip(shape_help_msg)
        stride_help_symbol.setObjectName('help_label')
        stride_help_symbol.setToolTip(stride_help_msg)
        spacing_help_symbol.setObjectName('help_label')
        spacing_help_symbol.setToolTip(spacing_help_msg)

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel(trans._('Grid stride:')), 0, 0)
        grid_layout.addWidget(grid_stride, 0, 1)
        grid_layout.addWidget(stride_help_symbol, 0, 2)
        grid_layout.addWidget(QLabel(trans._('Grid width:')), 1, 0)
        grid_layout.addWidget(grid_width, 1, 1)
        grid_layout.addWidget(shape_help_symbol, 1, 2, 2, 1)
        grid_layout.addWidget(QLabel(trans._('Grid height:')), 2, 0)
        grid_layout.addWidget(grid_height, 2, 1)
        grid_layout.addWidget(QLabel(trans._('Grid spacing:')), 3, 0)
        grid_layout.addWidget(grid_spacing, 3, 1)
        grid_layout.addWidget(spacing_help_symbol, 3, 2)

        popup.frame.setLayout(grid_layout)
        popup.show_above_mouse()

    def _update_grid_width(self, value):
        self.viewer.grid.shape = (self.viewer.grid.shape[0], value)

    def _update_grid_stride(self, value):
        self.viewer.grid.stride = value

    def _update_grid_height(self, value):
        self.viewer.grid.shape = (value, self.viewer.grid.shape[1])

    def _update_grid_spacing(self, value: float) -> None:
        self.viewer.grid.spacing = value

# --- End QtViewportButtons Class ---
