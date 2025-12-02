from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
import napari.layers

class LabelInfoWidget(QFrame):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        # UI Setup
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        self.info_label = QLabel("<b>No Label Selected</b>")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.setLayout(layout)

        # Connections
        self.viewer.layers.selection.events.active.connect(self._on_layer_change)
        self._on_layer_change(None) # Initialize

    def _on_layer_change(self, event):
        """Switch connection to the new active layer."""
        layer = self.viewer.layers.selection.active

        # Disconnect from old layers if necessary (omitted for brevity,
        # but in production you might want to track and disconnect old events)

        if isinstance(layer, napari.layers.Labels):
            # Connect to the label selection event
            layer.events.selected_label.connect(self._update_info)
            # Update immediately
            self._update_info(None)
        else:
            self.info_label.setText("<i>Select a Labels layer to see details.</i>")

    def _update_info(self, event):
        """Lookup properties for the selected label."""
        layer = self.viewer.layers.selection.active
        if not layer: return

        idx = layer.selected_label

        # Default text
        text = f"<b>Index:</b> {idx}"

        # 1. Check for 'properties' (dictionary of lists)
        if hasattr(layer, 'properties') and 'label' in layer.properties:
            labels = layer.properties['label']
            if idx in labels:
                # Find the index in the list where label == idx
                # Note: This assumes labels are unique.
                i = list(labels).index(idx)

                # Iterate over other keys in properties to display them
                for key, values in layer.properties.items():
                    if key == 'label': continue
                    if i < len(values):
                        text += f"<br><b>{key.capitalize()}:</b> {values[i]}"

        # 2. Check for 'features' (Pandas DataFrame - common in newer plugins)
        elif hasattr(layer, 'features') and not layer.features.empty:
            df = layer.features
            # Assuming there is a column 'label' or using the index
            if 'label' in df.columns:
                row = df[df['label'] == idx]
            else:
                row = df[df.index == idx]

            if not row.empty:
                for col in row.columns:
                    if col == 'label': continue
                    text += f"<br><b>{col.capitalize()}:</b> {row.iloc[0][col]}"

        self.info_label.setText(text)