import sys
import numpy as np
import napari
from PyQt6.QtWidgets import QLabel
from napari.layers import Labels, Image
from napari.utils.events import Event

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QSplitter, QHBoxLayout, QFileDialog, QAction
)
from qtpy.QtCore import Qt

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

        self._blk_mode = Blocker()
        self._blk_visibility = Blocker()
        self._blk_blending = Blocker()
        self._blk_affine = Blocker()

        self._blk_colormap = Blocker()
        self._blk_brush = Blocker()
        self._blk_n_edit_dimensions = Blocker()
        self._blk_selected_label = Blocker()
        self._blk_show_sel_label = Blocker()


        # --- 1. Initialize Viewers ---
        # We use the Axial viewer as the "Master" logic engine
        self.v_axial = napari.Viewer(title="Axial", show=False)
        self.v_coronal = napari.Viewer(title="Coronal", show=False)
        self.v_sagittal = napari.Viewer(title="Sagittal", show=False)
        self.v_3d = napari.Viewer(title="3D", show=False)

        self.viewers = [self.v_axial, self.v_coronal, self.v_sagittal, self.v_3d]

        # --- 2. Build UI (Menus & Layout) ---
        self._setup_menus()
        self._build_layout()

        # --- 3. Setup Data ---
        self._create_and_add_data()

        # --- 4. Setup Synchronization ---
        self._link_viewers_layer_list()
        self._link_viewer_tools()
        self._sync_active_selection()

        # Link existing label layers
        label_layers = [v.layers["Labels"] for v in self.viewers if "Labels" in v.layers]
        if label_layers:
            self._link_label_painting(label_layers)

        # --- 5. Configure Cameras ---
        self._setup_camera_orientations()

        # --- 6. UI Tweaks (Disable Transform) ---
        # We connect this LAST so it triggers on initial selection
        self.v_axial.layers.selection.events.active.connect(self._disable_transform_tool)
        self._disable_transform_tool(None) # Trigger once manually

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

        # Add the extracted widgets
        side_layout.addWidget(dock_controls)
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
        c = 64
        label_vol[c-10:c+10, c-10:c+10, c-10:c+10] = 1

        for v in self.viewers:
            v.add_image(image_vol, name="Volume", colormap="gray", contrast_limits=[0, 1])
            v.add_labels(label_vol, name="Labels", opacity=0.6)

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
        sync_layer_funcs = []
        sync_label_funcs = []


        @self._blk_mode
        def sync_mode(event:Event):
            source_layer = event.source
            mode = source_layer.mode
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.visible != mode:
                        corresponding_layer.mode = mode
                except KeyError:
                    pass


        @self._blk_visibility
        def sync_visibility(event:Event):
            source_layer = event.source
            visible = source_layer.visible
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.visible != visible:
                        corresponding_layer.blending = visible
                except KeyError:
                    pass

        @self._blk_blending
        def sync_blending(event:Event):
            source_layer = event.source
            blending = source_layer.blending
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.blending != blending:
                        corresponding_layer.blending = blending
                except KeyError:
                    pass

        @self._blk_affine
        def sync_affine(event:Event):
            source_layer = event.source
            affine = source_layer.affine
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.affine != affine:
                        corresponding_layer.affine = affine
                except KeyError:
                    pass

        @self._blk_colormap
        def sync_colormap(event:Event):
            source_layer = event.source
            colormap = source_layer.colormap
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.colormap != colormap:
                        corresponding_layer.colormap = colormap
                except KeyError:
                    pass

        @self._blk_brush
        def sync_brush(event):
            """Syncs the brush size across all viewers."""
            source_layer = event.source
            new_size = source_layer.brush_size
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.brush_size != new_size:
                        corresponding_layer.brush_size = new_size
                except KeyError:
                    pass

        @self._blk_n_edit_dimensions
        def sync_n_edit_dimensions(event:Event):
            source_layer = event.source
            n_edit_dimensions = source_layer.n_edit_dimensions
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.n_edit_dimensions != n_edit_dimensions:
                        corresponding_layer.n_edit_dimensions = n_edit_dimensions
                except KeyError:
                    pass

        @self._blk_selected_label
        def sync_selected_label(event: Event):
            """Syncs the selected label ID across all viewers."""
            source_layer = event.source
            new_label = source_layer.selected_label
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.selected_label != new_label:
                        corresponding_layer.selected_label = new_label
                except KeyError:
                    pass


        @self._blk_show_sel_label
        def sync_show_selected_label(event:Event):
            source_layer = event.source
            show_selected_label = source_layer.show_selected_label
            for v in self.viewers:
                try:
                    corresponding_layer = v.layers[source_layer.name]
                    if corresponding_layer and corresponding_layer.show_selected_label != show_selected_label:
                        corresponding_layer.show_selected_label = show_selected_label
                except KeyError:
                    pass

        def connect_layer(layer):
            layer.events.mode.connect(sync_mode)
            layer.events.visible.connect(sync_visibility)
            layer.events.blending.connect(sync_blending)
            layer.events.affine.connect(sync_affine)

            if isinstance(layer, Labels):
                layer.events.colormap.connect(sync_colormap)
                layer.events.brush_size.connect(sync_brush)
                layer.events.n_edit_dimensions.connect(sync_n_edit_dimensions)
                layer.events.show_selected_label.connect(sync_show_selected_label)
                layer.events.selected_label.connect(sync_selected_label)

        for v in self.viewers:
            for layer in v.layers:
                connect_layer(layer)
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
        for v in self.viewers:
            v.close()
        super().closeEvent(event)

# =============================================================================
# 3. Entry Point
# =============================================================================

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    from napari.utils.theme import get_theme
    from napari._qt.qt_resources import get_stylesheet
    theme = get_theme("dark")
    app.setStyleSheet(get_stylesheet(theme.id))

    window = MedicalMainWindow()
    window.show()
    sys.exit(app.exec_())