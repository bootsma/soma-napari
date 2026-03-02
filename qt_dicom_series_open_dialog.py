import logging
import os
import time
from typing import TypedDict, List, Dict

import numpy as np
import pydicom
import SimpleITK as sitk
from PyQt6.QtCore import (Qt, QObject, QThread, pyqtSignal)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QCheckBox
)

from mira_core.dicom_utils import get_dicom_series_info_dict
from mira_core.volume_info import VolumeInformation
from qt_theme_utils import (
    make_button, NAPARI_AVAILABLE
)

class DicomSeriesData(TypedDict):
    series_info: dict
    volume_info: VolumeInformation
    image_data: np.ndarray

class ScanDicomWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory

    def run(self):
        try:
            series_dict = get_dicom_series_info_dict(self.directory)
            self.finished.emit(series_dict)
        except Exception as e:
            self.error.emit(str(e))

class ReadDicomSeriesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open DICOM Series")
        self.resize(800, 600)

        self.discovered_series = {}
        self.selected_data: List[DicomSeriesData] = []
        self.clear_layers_on_load = True

        self._setup_ui()
        self.thread = None
        self.worker = None

    def _setup_ui(self):
        layout = QVBoxLayout()

        # --- Directory Selection ---
        dir_layout = QHBoxLayout()
        self.txt_dir_path = QLineEdit()
        self.txt_dir_path.setPlaceholderText("Select a folder containing DICOM files...")
        self.btn_browse_dir = make_button("folder", "Browse for DICOM Directory", self._browse_directory)
        dir_layout.addWidget(QLabel("Directory:"))
        dir_layout.addWidget(self.txt_dir_path)
        dir_layout.addWidget(self.btn_browse_dir)
        layout.addLayout(dir_layout)

        # --- Main Content Area (Splitter-like layout) ---
        content_layout = QHBoxLayout()

        # Left: Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["", "Description", "Dimensions", "Slices", "Series UID"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        content_layout.addWidget(self.table, stretch=2)

        # Right: Preview
        preview_panel = QVBoxLayout()
        self.lbl_preview = QLabel("Select a series to preview")
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

    def _browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if dir_path:
            self.txt_dir_path.setText(dir_path)
            self._start_scan(dir_path)

    def _start_scan(self, directory):
        self.table.setRowCount(0)
        self.btn_load.setEnabled(False)
        self.lbl_preview.setText("Scanning...")
        
        self.thread = QThread()
        self.worker = ScanDicomWorker(directory)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_scan_finished)
        self.worker.error.connect(self._on_scan_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_scan_finished(self, series_dict):
        self.discovered_series = series_dict
        self.table.setRowCount(len(series_dict))
        
        for row, (uid, info) in enumerate(series_dict.items()):
            # Checkbox for multiple selection
            check_widget = QWidget()
            check_layout = QHBoxLayout(check_widget)
            checkbox = QCheckBox()
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self._update_load_button_state)
            check_layout.addWidget(checkbox)
            check_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            check_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 0, check_widget)
            
            self.table.setItem(row, 1, QTableWidgetItem(info['description']))
            self.table.setItem(row, 2, QTableWidgetItem(info['dimensions']))
            self.table.setItem(row, 3, QTableWidgetItem(str(info['num_images'])))
            self.table.setItem(row, 4, QTableWidgetItem(info['series_instance_uid']))
            
            # Store UID in user data of the first item
            self.table.item(row, 1).setData(Qt.ItemDataRole.UserRole, uid)

        if not series_dict:
            self.lbl_preview.setText("No DICOM series found")
        else:
            self.lbl_preview.setText("Select a series to preview")

    def _on_scan_error(self, error_msg):
        QMessageBox.critical(self, "Scan Error", f"Failed to scan directory: {error_msg}")
        self.lbl_preview.setText("Scan failed")

    def _update_load_button_state(self):
        any_checked = False
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox.isChecked():
                any_checked = True
                break
        self.btn_load.setEnabled(any_checked)

    def _on_selection_changed(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        uid = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        info = self.discovered_series[uid]
        
        self._load_preview(info)

    def _load_preview(self, info):
        file_paths = info['file_paths']
        if not file_paths:
            return
        
        try:
            # Load middle slice
            mid_idx = len(file_paths) // 2
            mid_file = file_paths[mid_idx]
            
            ds = pydicom.dcmread(mid_file)
            if not hasattr(ds, 'pixel_array'):
                self.lbl_preview.setText("No Pixels")
                return

            arr = ds.pixel_array.astype(float)
            
            # Simple min/max normalization for display
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
            else:
                arr = np.zeros_like(arr)
            arr = arr.astype(np.uint8)

            height, width = arr.shape
            bytes_per_line = width
            q_img = QImage(arr.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            self.lbl_preview.setPixmap(pixmap.scaled(
                self.lbl_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            self.lbl_info.setText(f"Patient: {info['patient_id']}"
                                 f"Description: {info['description']}"
                                 f"Dimensions: {info['dimensions']} x {info['num_images']}")

        except Exception as e:
            logging.error(f"Error loading preview: {e}")
            self.lbl_preview.setText("Preview Error")

    def _on_load_clicked(self):
        self.selected_data = []
        try:
            for row in range(self.table.rowCount()):
                checkbox_widget = self.table.cellWidget(row, 0)
                if checkbox_widget:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox and checkbox.isChecked():
                        uid = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
                        info = self.discovered_series[uid]

                        # Load the actual volume using sitk
                        reader = sitk.ImageSeriesReader()
                        reader.SetFileNames(info['file_paths'])
                        reader.MetaDataDictionaryArrayUpdateOn()
                        image = reader.Execute()

                        vol_info = VolumeInformation.from_sitk_image(image)
                        image_data = sitk.GetArrayFromImage(image)

                        self.selected_data.append({
                            "series_info": info,
                            "volume_info": vol_info,
                            "image_data": image_data
                        })
            if not self.selected_data:
                QMessageBox.warning(self, "No Selection", "Please select at least one series to load.")
                return

            # Store the checkbox state
            self.clear_layers_on_load = self.chk_clear_layers.isChecked()

            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load selected series: {e}")
