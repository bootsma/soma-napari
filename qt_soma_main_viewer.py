import sys
import numpy as np
import napari
from PyQt6.QtWidgets import QLabel
from napari.layers import Labels, Image

from napari._qt.dialogs.qt_activity_dialog import QtActivityDialog
from napari._qt.widgets.qt_viewer_status_bar import ViewerStatusBar
from napari._qt.threads.status_checker import StatusChecker


from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QSplitter, QFileDialog, QAction
)
from qtpy.QtCore import Qt

from qt_label_info_widget import LabelInfoWidget
from qt_theme_utils import copy_custom_ui_icons, customize_stylesheet

# --- Import your custom buttons if available ---
try:
    from qt_viewport import QtViewport
    HAS_VIEWPORT = True
except ImportError:
    print("Error: qt_viewport.py not found.")
    HAS_VIEWPORT = False

# =============================================================================
# 1. Utility Classes
# =============================================================================

class Blocker:
    """Context manager to prevent infinite event recursion."""
    def __init__(self):
        self._busy = False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self._busy:
                return
            self._busy = True
            try:
                func(*args, **kwargs)
            finally:
                self._busy = False
        return wrapper

# =============================================================================
# 2. Main Window Class
# =============================================================================

class MedicalMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soma Napari Medical Viewer")
        self.resize(1400, 900)

        # --- Define Blockers ---
        self._blk_layer_insert = Blocker()
        self._blk_layer_remove = Blocker()

        self._blk_active = Blocker()

        # We use the Axial viewer as the "Master" logic engine
        self.v_axial = napari.Viewer(title="Axial", show=False)
        self.v_coronal = napari.Viewer(title="Coronal", show=False)
        self.v_sagittal = napari.Viewer(title="Sagittal", show=False)
        self.v_3d = napari.Viewer(title="3D", show=False)

        self.viewers = [self.v_axial, self.v_coronal, self.v_sagittal, self.v_3d]

        target_widget = self.v_axial.window._qt_viewer._welcome_widget
        self._activity_dialog = QtActivityDialog(target_widget)
        self._activity_dialog.hide()

        # Keep the dialog positioned correctly when the window resizes
        target_widget.resized.connect(self._activity_dialog.move_to_bottom_right)

        # Setup Status Bar
        self.status_bar = ViewerStatusBar(self)
        self.setStatusBar(self.status_bar)

        # Keep track of status checker threads
        self.status_checkers = []

        # Connect listeners (as defined in the previous answer)
        for viewer in self.viewers:
            self._attach_status_management(viewer)

        self._create_and_add_data()
        self._setup_camera_orientations()

        # ---  Setup Data ---
        self._setup_menus()
        self._build_layout()

        #  Setup Synchronization ---
        self._link_viewers_layer_list()
        self._link_viewer_tools()
        self._sync_active_selection()

        # Link existing label layers
        label_layers = [v.layers["Labels"] for v in self.viewers if "Labels" in v.layers]
        if label_layers:
            self._link_label_painting(label_layers)

        #  UI Tweaks (Disable Transform) ---
        # We connect this LAST so it triggers on initial selection
        self.v_axial.layers.selection.events.active.connect(self._disable_transform_tool)
        self._disable_transform_tool(None) # Trigger once manually


    def _attach_status_management(self, viewer):
        """
        Creates a StatusChecker thread for a viewer and links
        events to the main window's status bar.
        """
        # A. Create the background thread
        checker = StatusChecker(viewer, parent=self)
        self.status_checkers.append(checker)

        # B. Define what happens when the checker finishes a calculation
        def apply_status_to_viewer(status_info):

            if status_info is None:
                return  # Do nothing (or clear status) if signal is empty

            # status_info is a tuple: (status_text, tooltip_text)
            viewer.status = status_info[0]
            viewer.tooltip.text = status_info[1]

        checker.status_and_tooltip_changed.connect(apply_status_to_viewer)

        # ... (Rest of the function remains the same) ...
        checker.status_and_tooltip_changed.connect(apply_status_to_viewer)
        viewer.cursor.events.position.connect(checker.trigger_status_update)
        viewer.events.status.connect(self._on_status_changed)
        viewer.events.help.connect(lambda e: self.status_bar.setHelpText(e.value))
        checker.start()

    def _on_status_changed(self, event):
        """Handle status updates from any viewer."""
        # Napari sends either a simple string or a dictionary of coordinate info
        if isinstance(event.value, str):
            self.status_bar.setStatusText(event.value)
        else:
            # Complex object (coordinates, layer values, etc)
            status_info = event.value
            self.status_bar.setStatusText(
                layer_base=status_info.get('layer_base'),
                source_type=status_info.get('source_type'),
                plugin=status_info.get('plugin'),
                coordinates=status_info.get('coordinates')
            )
    # -------------------------------------------------------------------------
    # GUI Construction
    # -------------------------------------------------------------------------




    def _setup_menus(self):
        """Creates a standard File/View menu bar."""
        menubar = self.menuBar()

        # -- File Menu --
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open File(s)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_files_dialog)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # -- View Menu --
        view_menu = menubar.addMenu("View")
        reset_action = QAction("Reset All Cameras", self)
        reset_action.triggered.connect(self._reset_all_cameras)
        view_menu.addAction(reset_action)

    def _open_files_dialog(self):
        """Opens files using Napari's internal reader."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Medical Image")
        if file_paths:
            # Open in the Master (Axial) viewer.
            # The Sync logic will automatically propagate layers to the others.
            self.v_axial.open(file_paths)

    def _reset_all_cameras(self):
        for v in self.viewers:
            v.reset_view()

    def _build_layout(self):
        """Constructs the 4-panel layout with sidebar."""
        # -- Viewports --
        if HAS_VIEWPORT:
            vp_ax = QtViewport(self.v_axial, "Axial (Z)", leave_controls=True)
            vp_co = QtViewport(self.v_coronal, "Coronal (Y)")
            vp_sa = QtViewport(self.v_sagittal, "Sagittal (X)")
            vp_3d = QtViewport(self.v_3d, "3D View")
        else:
            # Fallback if import fails
            vp_ax = QLabel("Viewport Error")

        # -- Splitters (2x2 Grid) --
        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(vp_ax)
        left_split.addWidget(vp_co)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(vp_sa)
        right_split.addWidget(vp_3d)

        view_split = QSplitter(Qt.Horizontal)
        view_split.addWidget(left_split)
        view_split.addWidget(right_split)


        axial_qt = self.v_axial.window._qt_viewer
        dock_controls = axial_qt.dockLayerControls
        dock_layers = axial_qt.dockLayerList

        try:
            main_viewer_buttons = axial_qt.viewerButtons #

            main_viewer_buttons.rollDimsButton.setVisible(False)
            main_viewer_buttons.transposeDimsButton.setVisible(False)
            main_viewer_buttons.resetViewButton.setVisible(False)
            main_viewer_buttons.gridViewButton.setVisible(False)
            main_viewer_buttons.ndisplayButton.setVisible(False)
            # We leave main_viewer_buttons.consoleButton visible!
        except Exception as e:
            print(f"Could not customize main viewer buttons: {e}")


        # -- Left Sidebar --
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout()
        side_layout.setContentsMargins(5, 5, 5, 5)


        # --- NEW CODE START ---
        # Create and add your info widget
        self.label_info_widget = LabelInfoWidget(self.v_axial)

        # Placing it between controls and layer list often looks best.
        side_layout.addWidget(dock_controls)
        side_layout.addWidget(self.label_info_widget)
        side_layout.addWidget(dock_layers)
        sidebar.setLayout(side_layout)


        # -- Main Splitter --
        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(sidebar)
        main_split.addWidget(view_split)

        # Set initial stretch factors
        main_split.setStretchFactor(0, 0) # Sidebar doesn't stretch
        main_split.setStretchFactor(1, 1) # Viewports take space

        self.setCentralWidget(main_split)

    def _disable_transform_tool(self, event):
        """
        Hides the 'Transform' button in the layer controls.
        Called whenever the active layer changes.
        """
        try:
            # 1. Get the Layer Controls Container
            qt_viewer = self.v_axial.window._qt_viewer
            controls_container = qt_viewer.dockLayerControls.widget() # QtLayerControlsContainer

            # 2. Get the specific controls widget for the active layer
            current_controls = controls_container.currentWidget()

            # 3. Find and hide the transform button
            # Note: Different layer types have different control classes,
            # but they all inherit from QtLayerControls which has 'transform_button'
            if hasattr(current_controls, 'transform_button'):
                current_controls.transform_button.setVisible(False)

        except Exception as e:
            pass # Fail silently if widget structure isn't ready

    # -------------------------------------------------------------------------
    # Standard Setup (Data & Cameras)
    # -------------------------------------------------------------------------

    def _create_and_add_data(self):
        dims = (128, 128, 128)
        image_vol = np.random.rand(*dims).astype(np.float32)
        label_vol = np.zeros(dims, dtype=np.int32)

        # Define some label indices
        c = 64
        label_vol[c-10:c+10, c-10:c+10, c-10:c+10] = 1 # Label 1
        label_vol[c-20:c-15, c-20:c-15, c-20:c-15] = 2 # Label 2

        # Define properties mapping
        # Ensure the 'label' list matches the integer values in label_vol
        label_properties = {
            'label': [1, 2],
            'name': ['Tumor Core', 'Edema']
        }

        for v in self.viewers:
            v.add_image(image_vol, name="Volume", colormap="gray", contrast_limits=[0, 1])

            # Add labels with properties
            lbl_layer = v.add_labels(
                label_vol,
                name="Labels",
                opacity=0.6,
                properties=label_properties # Pass the dictionary here
            )

    def _setup_camera_orientations(self):
        center = (64, 64, 64)
        self.v_axial.dims.ndisplay = 2
        self.v_axial.dims.order = (0, 1, 2)
        self.v_axial.dims.set_point(0, center[0])

        self.v_coronal.dims.ndisplay = 2
        self.v_coronal.dims.order = (1, 0, 2)
        self.v_coronal.dims.set_point(1, center[1])

        self.v_sagittal.dims.ndisplay = 2
        self.v_sagittal.dims.order = (2, 0, 1)
        self.v_sagittal.dims.set_point(2, center[2])

        self.v_3d.dims.ndisplay = 3
        self.v_3d.camera.angles = (0, 0, 90)

    # -------------------------------------------------------------------------
    # Synchronization Logic (Refined)
    # -------------------------------------------------------------------------

    def _link_viewers_layer_list(self):
        @self._blk_layer_insert
        def on_insert(event):
            layer = event.value
            source_viewer = next((v for v in self.viewers if v.layers is event.source), None)
            if not source_viewer: return

            # Add to others
            for v in self.viewers:
                if v is not source_viewer and layer.name not in v.layers:
                    if isinstance(layer, Labels):
                        new_l = v.add_labels(layer.data, name=layer.name, opacity=layer.opacity)
                        # IMPORTANT: Link painting for the new layer immediately
                        all_labels = [vv.layers[layer.name] for vv in self.viewers if layer.name in vv.layers]
                        self._link_label_painting(all_labels)
                    elif isinstance(layer, Image):
                        v.add_image(layer.data, name=layer.name, colormap=layer.colormap.name, blending=layer.blending)

        @self._blk_layer_remove
        def on_remove(event):
            layer = event.value
            for v in self.viewers:
                if layer.name in v.layers:
                    v.layers.remove(layer.name)

        for v in self.viewers:
            v.layers.events.inserted.connect(on_insert)
            v.layers.events.removed.connect(on_remove)

    def _sync_active_selection(self):
        @self._blk_active
        def on_active_change(event):
            new_active = event.value
            if new_active is None: return
            for v in self.viewers:
                if new_active.name in v.layers:
                    v.layers.selection.active = v.layers[new_active.name]

            # Trigger UI refresh for the transform button
            self._disable_transform_tool(None)

        for v in self.viewers:
            v.layers.selection.events.active.connect(on_active_change)

    def _link_viewer_tools(self):
        from functools import partial

        # 1. Configuration: Define what to sync
        # Format: 'attribute_name' (assumes event name is same as attribute name)
        # If event name differs, we could use a tuple, but Napari is usually consistent.

        # Attributes common to all Layers
        base_attributes = [
            'mode',
            'visible',
            'blending',
            'affine',
            'opacity' # Added opacity as it's commonly needed
        ]

        image_attributes = [
            'contrast_limits',
            'gamma',
            'interpolation2d',
            'interpolation3d',
            'colormap'
        ]

        # Attributes specific to Labels layers
        label_attributes = [
            'colormap',
            'brush_size',
            'n_edit_dimensions',
            'show_selected_label',
            'selected_label',
            'contour', # Example of how easy it is to add new props
            'preserve_labels'
        ]

        # 2. State tracking to prevent infinite recursion
        # We replace the explicit @Blocker decorators with a set of active keys.
        self._active_sync_keys = set()

        # 3. The Generic Sync Handler
        def _sync_attribute(event, attr_name):
            # Create a unique key for this attribute to prevent recursion
            # We use attr_name so syncing 'opacity' doesn't block syncing 'brush_size'
            if attr_name in self._active_sync_keys:
                return

            self._active_sync_keys.add(attr_name)
            try:
                source_layer = event.source

                # Get the value dynamically
                # We use getattr ensure we have the current state,
                # though event.value often holds it.
                new_value = getattr(source_layer, attr_name)

                for v in self.viewers:
                    # Skip if the layer doesn't exist in this viewer
                    if source_layer.name not in v.layers:
                        continue

                    target_layer = v.layers[source_layer.name]

                    # Optimization: Only write if values differ
                    # This also acts as a secondary recursion guard
                    current_value = getattr(target_layer, attr_name)

                    # Handle numpy array comparisons if necessary, otherwise standard equality
                    are_different = False
                    if isinstance(new_value, np.ndarray):
                        are_different = not np.array_equal(new_value, current_value)
                    else:
                        are_different = new_value != current_value

                    if are_different:
                        setattr(target_layer, attr_name, new_value)

            except AttributeError:
                # Handle cases where a layer might not have the attribute
                pass
            except KeyError:
                pass
            finally:
                self._active_sync_keys.remove(attr_name)

        # 4. Connection Helper
        def connect_layer(layer):
            # Connect Base Attributes
            for attr in base_attributes:
                if hasattr(layer.events, attr):
                    # We use partial to pass the specific attr_name to the generic function
                    getattr(layer.events, attr).connect(
                        partial(_sync_attribute, attr_name=attr)
                    )

            # Connect Label Specific Attributes
            if isinstance(layer, Labels):
                for attr in label_attributes:
                    if hasattr(layer.events, attr):
                        getattr(layer.events, attr).connect(
                            partial(_sync_attribute, attr_name=attr)
                        )
            elif isinstance(layer, Image):
                for attr in image_attributes:
                    if hasattr(layer.events, attr):
                        getattr(layer.events, attr).connect(
                            partial(_sync_attribute, attr_name=attr)
                        )

        # 5. Apply to existing and future layers
        for v in self.viewers:
            for layer in v.layers:
                connect_layer(layer)
            # Connect newly inserted layers automatically
            v.layers.events.inserted.connect(lambda e: connect_layer(e.value))


    def _link_label_painting(self, label_layers):
        def create_vis_sync(source, targets):
            def on_update(event):
                if event.source is source:
                    for t in targets: t.refresh()
            return on_update

        def create_hist_sync(targets):
            def on_paint(event):
                for t in targets:
                    t._undo_history.append(event.value)
                    t._redo_history.clear()
            return on_paint

        self._monkey_patch_undo_redo(label_layers)

        for i, source in enumerate(label_layers):
            others = [L for j, L in enumerate(label_layers) if i != j]
            source.events.labels_update.connect(create_vis_sync(source, others))
            source.events.paint.connect(create_hist_sync(others))

    def _monkey_patch_undo_redo(self, layers):
        is_undoing = [False]
        is_redoing = [False]

        for layer in layers:
            if not hasattr(layer, '_original_undo'):
                layer._original_undo = layer.undo
            if not hasattr(layer, '_original_redo'):
                layer._original_redo = layer.redo

        def synced_undo():
            if is_undoing[0]: return
            is_undoing[0] = True
            try:
                for l in layers: l._original_undo()
            finally:
                is_undoing[0] = False

        def synced_redo():
            if is_redoing[0]: return
            is_redoing[0] = True
            try:
                for l in layers: l._original_redo()
            finally:
                is_redoing[0] = False

        for layer in layers:
            layer.undo = synced_undo
            layer.redo = synced_redo

    def closeEvent(self, event):

        for checker in self.status_checkers:
            checker.close_terminate()
            checker.wait()3
            v.close()
        super().closeEvent(event)





if __name__ == "__main__":

    copy_custom_ui_icons()
    app = QApplication.instance() or QApplication(sys.argv)

    window = MedicalMainWindow()
    customize_stylesheet(app)


    window.show()
    sys.exit(app.exec_())