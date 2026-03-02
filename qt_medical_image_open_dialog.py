import logging
import os
from typing import TypedDict, List, Dict

import numpy as np
import SimpleITK as sitk
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QCheckBox, QComboBox
)

from mira_core.volume_info import VolumeInformation
from qt_theme_utils import make_button, NAPARI_AVAILABLE


class MedicalImageData(TypedDict):
    file_path: str
    volume_info: VolumeInformation
    image_data: np.ndarray
    data_type: str  # "Image" or "Segmentation"


class AnalyzeFileWorker(QObject):
    """Worker thread for analyzing medical image files and extracting metadata."""
    finished = pyqtSignal(list)  # List of dicts with file metadata
    error = pyqtSignal(str)

    def __init__(self, file_paths: List[str]):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        try:
            results = []
            for file_path in self.file_paths:
                try:
                    # Read image metadata only
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(file_path)
                    reader.ReadImageInformation()

                    size = reader.GetSize()
                    spacing = reader.GetSpacing()

                    # Detect data type
                    detected_type = self._detect_type(reader, file_path)

                    results.append({
                        'file_path': file_path,
                        'filename': os.path.basename(file_path),
                        'dimensions': f"{size[0]}x{size[1]}x{size[2]}",
                        'spacing': spacing,
                        'size': size,
                        'detected_type': detected_type
                    })
                except Exception as e:
                    logging.warning(f"Could not read {file_path}: {e}")
                    continue

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def _detect_type(self, reader: sitk.ImageFileReader, file_path: str) -> str:
        """
        Detect if the file is likely an Image or Segmentation.

        Heuristics:
        1. Check filename for keywords (seg, label, mask)
        2. Check pixel type (integer types more likely segmentation)
        3. Default to Image
        """
        filename = os.path.basename(file_path).lower()

        # Check filename
        seg_keywords = ['seg', 'label', 'mask', 'contour', 'annotation']
        if any(keyword in filename for keyword in seg_keywords):
            return "Segmentation"

        # Check pixel type
        pixel_type = reader.GetPixelID()
        # Integer types: sitkUInt8, sitkInt8, sitkUInt16, sitkInt16, sitkUInt32, sitkInt32
        integer_types = [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16,
                        sitk.sitkInt16, sitk.sitkUInt32, sitk.sitkInt32]

        if pixel_type in integer_types:
            # Could be segmentation, but not definitive
            # Default to Image unless filename suggests otherwise
            pass

        # Default to Image
        return "Image"


class ReadMedicalImageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Medical Images")
        self.resize(900, 600)

        self.file_metadata = []
        self.selected_data: List[MedicalImageData] = []
        self.clear_layers_on_load = False

        self._setup_ui()
        self.thread = None
        self.worker = None

    def _setup_ui(self):
        layout = QVBoxLayout()

        # --- File Selection ---
        file_layout = QHBoxLayout()
        self.txt_file_paths = QLineEdit()
        self.txt_file_paths.setPlaceholderText("Select medical image files...")
        self.txt_file_paths.setReadOnly(True)
        self.btn_browse_files = make_button("folder", "Browse for Files", self._browse_files)
        file_layout.addWidget(QLabel("Files:"))
        file_layout.addWidget(self.txt_file_paths)
        file_layout.addWidget(self.btn_browse_files)
        layout.addLayout(file_layout)

        # --- Main Content Area ---
        content_layout = QHBoxLayout()

        # Left: Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["", "Filename", "Dimensions", "Type", "Path"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        content_layout.addWidget(self.table, stretch=2)

        # Right: Preview
        preview_panel = QVBoxLayout()
        self.lbl_preview = QLabel("Select files to preview")
        self.lbl_preview.setFixedSize(256, 256)
        self.lbl_preview.setStyleSheet("border: 1px solid #555; background-color: #000;")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_panel.addWidget(self.lbl_preview)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        preview_panel.addWidget(self.lbl_info)
        preview_panel.addStretch()

        content_layout.addLayout(preview_panel, stretch=1)
        layout.addLayout(content_layout)

        # --- Options ---
        options_layout = QHBoxLayout()
        self.chk_clear_layers = QCheckBox("Clear existing layers before loading")
        self.chk_clear_layers.setChecked(False)
        options_layout.addWidget(self.chk_clear_layers)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_load = QPushButton("Load Selected")
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_load.setEnabled(False)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _browse_files(self):
        """Open file dialog to select medical image files."""
        file_filter = (
            "Medical Images ("
            "*.nii *.nii.gz "
            "*.mha *.mhd "
            "*.nrrd *.nhdr "
            "*.mnc "
            "*.hdr *.img"
            ");;"
            "NIfTI (*.nii *.nii.gz);;"
            "MetaImage (*.mha *.mhd);;"
            "NRRD (*.nrrd *.nhdr);;"
            "MINC (*.mnc);;"
            "Analyze (*.hdr *.img);;"
            "All Files (*)"
        )

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Medical Image Files",
            "",
            file_filter
        )

        if file_paths:
            self.txt_file_paths.setText(f"{len(file_paths)} file(s) selected")
            self._start_analysis(file_paths)

    def _start_analysis(self, file_paths):
        """Start background analysis of selected files."""
        self.table.setRowCount(0)
        self.btn_load.setEnabled(False)
        self.lbl_preview.setText("Analyzing files...")

        self.thread = QThread()
        self.worker = AnalyzeFileWorker(file_paths)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_analysis_finished(self, results):
        """Populate table with analyzed file metadata."""
        self.file_metadata = results
        self.table.setRowCount(len(results))

        for row, metadata in enumerate(results):
            # Checkbox for selection
            check_widget = QWidget()
            check_layout = QHBoxLayout(check_widget)
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Auto-select all by default
            checkbox.stateChanged.connect(self._update_load_button_state)
            check_layout.addWidget(checkbox)
            check_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            check_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 0, check_widget)

            # Filename
            self.table.setItem(row, 1, QTableWidgetItem(metadata['filename']))

            # Dimensions
            self.table.setItem(row, 2, QTableWidgetItem(metadata['dimensions']))

            # Type (editable combo box)
            type_combo = QComboBox()
            type_combo.addItems(["Image", "Segmentation"])
            type_combo.setCurrentText(metadata['detected_type'])
            self.table.setCellWidget(row, 3, type_combo)

            # Full path
            path_item = QTableWidgetItem(metadata['file_path'])
            self.table.setItem(row, 4, path_item)

            # Store metadata in user data
            self.table.item(row, 1).setData(Qt.ItemDataRole.UserRole, metadata)

        if not results:
            self.lbl_preview.setText("No valid files found")
        else:
            self.lbl_preview.setText("Select a file to preview")
            self._update_load_button_state()

    def _on_analysis_error(self, error_msg):
        QMessageBox.critical(self, "Analysis Error", f"Failed to analyze files: {error_msg}")
        self.lbl_preview.setText("Analysis failed")

    def _update_load_button_state(self):
        """Enable load button if any files are checked."""
        any_checked = False
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                any_checked = True
                break
        self.btn_load.setEnabled(any_checked)

    def _on_selection_changed(self):
        """Update preview when user selects a different row."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        metadata = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)

        self._load_preview(metadata)

    def _load_preview(self, metadata):
        """Load and display middle slice preview of selected file."""
        try:
            file_path = metadata['file_path']

            # Read the full image for preview
            image = sitk.ReadImage(file_path)
            arr = sitk.GetArrayFromImage(image)

            # Get middle slice (remember: arr is ZYX)
            mid_z = arr.shape[0] // 2
            mid_slice = arr[mid_z, :, :]

            # Normalize for display
            if mid_slice.max() > mid_slice.min():
                normalized = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min()) * 255
            else:
                normalized = np.zeros_like(mid_slice)

            display_arr = normalized.astype(np.uint8)

            # Create QImage
            height, width = display_arr.shape
            bytes_per_line = width
            q_img = QImage(display_arr.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

            # Display
            pixmap = QPixmap.fromImage(q_img)
            self.lbl_preview.setPixmap(pixmap.scaled(
                self.lbl_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            # Update info
            spacing = metadata['spacing']
            self.lbl_info.setText(
                f"File: {metadata['filename']}\n"
                f"Dimensions: {metadata['dimensions']}\n"
                f"Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm"
            )

        except Exception as e:
            logging.error(f"Error loading preview: {e}")
            self.lbl_preview.setText("Preview Error")
            self.lbl_info.setText(f"Error: {e}")

    def _on_load_clicked(self):
        """Load selected files with their specified types."""
        self.selected_data = []
        try:
            for row in range(self.table.rowCount()):
                checkbox_widget = self.table.cellWidget(row, 0)
                if checkbox_widget:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox and checkbox.isChecked():
                        metadata = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
                        type_combo = self.table.cellWidget(row, 3)
                        selected_type = type_combo.currentText()

                        # Load the actual volume
                        file_path = metadata['file_path']
                        image = sitk.ReadImage(file_path)
                        vol_info = VolumeInformation.from_sitk_image(image)
                        image_data = sitk.GetArrayFromImage(image)

                        self.selected_data.append({
                            "file_path": file_path,
                            "volume_info": vol_info,
                            "image_data": image_data,
                            "data_type": selected_type
                        })

            if not self.selected_data:
                QMessageBox.warning(self, "No Selection", "Please select at least one file to load.")
                return

            # Store the checkbox state
            self.clear_layers_on_load = self.chk_clear_layers.isChecked()

            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load selected files: {e}")
