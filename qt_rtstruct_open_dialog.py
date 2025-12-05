import logging


import sys
import os
import time

from mira_core.dicom_utils import get_ref_image_series_uid, get_unique_series_uids, get_rtstruct_roi_names
from mira_core.contour import RtStructCombinedMaskLoader
from pydicom.errors import InvalidDicomError

import numpy as np
import pydicom
from typing import TypedDict, List, Dict


from PyQt6.QtGui import QPalette, QColor, QPixmap, QImage
from PyQt6.QtCore import ( Qt, QObject, QThread, pyqtSignal)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QFileDialog, QWidget, QTextEdit,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QMessageBox, QProgressBar
)

from mira_core.volume_info import VolumeInformation
from qt_theme_utils import (
    copy_custom_ui_icons, customize_stylesheet,
    make_button, NAPARI_AVAILABLE
)

# --- Napari Theme Imports ---
if NAPARI_AVAILABLE:
    try:
        from napari.utils.theme import get_theme
        from napari._qt.qt_resources import get_stylesheet
    except ImportError:
        pass

# --- Types & Constants ---

class DICOMRTStructData(TypedDict):
    dicom_rtstruct_file: str
    dicom_image_set_dir: str
    ref_image_series_uid: str
    roi_list: list

class DicomVolumeData(TypedDict):
    image_info: VolumeInformation
    image_data: np.ndarray
    mask_data: np.ndarray
    warnings: str
    error: str

DEFAULT_ROI_FILTER ={
    'filter_name':'GynSegmentation',
    'rois':[
        {
            'name':'Bladder',
            'index':1,
            'include_keys':['bladder'],
            'exclude_keys':[]
        },
        {
            'name':'Rectum',
            'index':2,
            'include_keys':['rectum']
        },
        {
            'name':'Sigmoid',
            'index':4,
            'include_keys':['sigmoid']
        },
        {
            'name':'SmallBowel',
            'index':3,
            'include_keys':['smallbowel', 'bowel']
        },
    ],
    'global_exclude':['fake', 'inside']
}

class ReadStructureDataWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)


    def __init__(self, requested_dicom_data: DICOMRTStructData, roi_index_map:dict ):
        super().__init__()
        self.requested = requested_dicom_data
        self.loaded_data = DicomVolumeData()
        self.roi_index_map = roi_index_map
        self.logger = logging.getLogger(__name__)

    def run(self):
        try:
            start = time.perf_counter()
            self.logger.info("Loading image data")
            image_info, image_data= VolumeInformation.get_volume_info_and_array(self.requested["dicom_rtstruct_file"],self.requested["dicom_image_set_dir"] )
            self.loaded_data["image_info"]=image_info
            self.loaded_data["image_data"]=image_data
            end =time.perf_counter()
            self.logger.info(f"Image Load took {end-start} seconds.")
            mask_data = RtStructCombinedMaskLoader(image_info, self.requested["dicom_rtstruct_file"],
                                                        self.roi_index_map,progress_callback=self.progress.emit)
            self.loaded_data["mask_data"] =mask_data
            end_mask = time.perf_counter()
            self.logger.info(f"Mask Load took {end_mask-end} seconds.")
        except Exception as e:
            msg = f"Failed to load data from {self.requested['dicom_rtstruct_file']}: {str(e)}"
            self.loaded_data["error"]=msg
        finally:
            self.finished.emit()

class ProcessingWindow(QDialog):
    """
    Step 2: ROI Mapping and Metadata Verification.
    """
    def __init__(self, dicom_data: DICOMRTStructData, filter_config: dict = None):
        super().__init__()
        self.setWindowTitle("ROI Segmentation Assignment")
        self.resize(400, 400) # Slightly wider for image preview

        self.dicom_data = dicom_data
        self.processed_data = None
        self.filter_config = filter_config or DEFAULT_ROI_FILTER
        self.available_rois = sorted(dicom_data.get('roi_list', []))
        self.combo_refs: List[QComboBox] = []
        self.current_assignments: List[str] = [] # Tracks text in each row to enable swapping

        self.setup_ui()
        # Thread management
        self.thread = None
        self.worker = None

        self.load_patient_data()
        self.load_preview_image()
        self.populate_table()

    def setup_ui(self):
        layout = QVBoxLayout()

        # --- 1. Patient Information Group ---
        self.info_group = QGroupBox("Patient Information")
        # Main layout for group is Horizontal: Image on Left, Text on Right
        group_layout = QHBoxLayout()

        # Left Side: Image Preview
        self.lbl_image_preview = QLabel("No Preview")
        self.lbl_image_preview.setFixedSize(150, 150)
        self.lbl_image_preview.setStyleSheet("border: 1px solid #555; background-color: #000;")
        self.lbl_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Right Side: Text Info
        self.info_layout = QFormLayout()
        self.lbl_name = QLabel("Loading...")
        self.lbl_id = QLabel("Loading...")
        self.lbl_date = QLabel("Loading...")
        self.lbl_modality = QLabel("Loading...")
        self.lbl_dims = QLabel("Loading...")
        self.lbl_proc_desc = QLabel("Loading...")

        self.info_layout.addRow("Patient Name:", self.lbl_name)
        self.info_layout.addRow("Patient ID:", self.lbl_id)
        self.info_layout.addRow("Study Date:", self.lbl_date)
        self.info_layout.addRow("Modality:", self.lbl_modality)
        self.info_layout.addRow("Volume Size:", self.lbl_dims)
        self.info_layout.addRow("Procedure:", self.lbl_proc_desc)

        # Add to group layout (Image first for Left side)
        group_layout.addWidget(self.lbl_image_preview)
        group_layout.addLayout(self.info_layout, stretch=1)

        self.info_group.setLayout(group_layout)
        layout.addWidget(self.info_group)

        # --- 2. ROI Mapping Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(3) # Increased to 3 columns
        self.table.setHorizontalHeaderLabels(["ID", "Target Structure", "DICOM Contour"])
        self.table.verticalHeader().setVisible(False)
        #self.table.setFixedHeight(100)

        # Header resizing
        header = self.table.horizontalHeader()

        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Target
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)          # Combo

        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self.table)

        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        layout.addWidget(self.pbar)

        # --- 3. Action Buttons ---
        btn_layout = QHBoxLayout()

        self.btn_back = QPushButton("Back")
        self.btn_back.setFixedWidth(100)
        self.btn_back.clicked.connect(self.reject)

        self.btn_process = QPushButton("Process")
        self.btn_process.setFixedWidth(100)
        self.btn_process.clicked.connect(self.process_data)

        btn_layout.addWidget(self.btn_back)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_process)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def process_data(self):
        try:
            self.pbar.setVisible(True)
            self.start_task()

        except Exception as e:
            print(f"Error: {e}")


    def start_task(self):
        # 1. Prepare Data

        roi_index_map, roi_target_map = self.get_roi_assignments()

        # 2. Create a QThread object
        self.thread = QThread()

        # 3. Create the Worker object
        self.worker = ReadStructureDataWorker(self.dicom_data, roi_index_map)

        # 4. Move worker to the thread
        self.worker.moveToThread(self.thread)

        # 5. Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Connect worker progress to UI updates
        #self.worker.error.connect(self.display_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.task_finished)

        # 6. Start the thread
        self.thread.start()

        # UI updates
        self.btn_back.setEnabled(False)
        self.btn_process.setEnabled(False)


    def update_progress(self, val):

        self.pbar.setValue(val)

    def display_error(self, message):
        QMessageBox.critical(self, "DICOM Data Error", message)


    def task_finished(self):
        #self.status_label.setText("Done!")
        if self.worker.loaded_data.get("error"):
            print("failed: ", self.worker.loaded_data["error"] )
            self.display_error(self.worker.loaded_data["error"])
            self.reject()
        else:
            self.processed_data = self.worker.loaded_data
            self.btn_process.setEnabled(True)
            self.accept()



    def load_patient_data(self):
        """Reads the DICOM file to extract patient metadata."""
        path = self.dicom_data.get('dicom_rtstruct_file')
        if not path:
            return

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)

            def get_val(tag, default="N/A"):
                val = getattr(ds, tag, default)
                return str(val) if val else default

            name = get_val('PatientName')
            if '^' in name:
                parts = name.split('^')
                name = f"{parts[1]} {parts[0]}" if len(parts) > 1 else parts[0]

            self.lbl_name.setText(name)
            self.lbl_id.setText(get_val('PatientID'))

            date = get_val('StudyDate')
            if len(date) == 8:
                date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
            self.lbl_date.setText(date)

        except Exception as e:
            self.lbl_name.setText("Error reading DICOM")
            print(f"Error reading patient info: {e}")

    def load_preview_image(self):
        """Loads metadata and a middle slice from the DICOM image directory."""
        image_dir = self.dicom_data.get('dicom_image_set_dir')
        if not image_dir or not os.path.exists(image_dir):
            return

        try:
            # 1. Find DICOM files
            files = [f for f in os.listdir(image_dir) if f.lower().endswith('.dcm')]
            if not files:
                self.lbl_image_preview.setText("No DICOMs")
                self.lbl_dims.setText("N/A")
                return

            # Count slices
            num_slices = len(files)

            # 2. Pick middle slice (naive sort by filename usually works for simple cases)
            files.sort()
            mid_idx = num_slices // 2
            dicom_path = os.path.join(image_dir, files[mid_idx])

            # 3. Read and Normalize
            ds = pydicom.dcmread(dicom_path)

            # --- Extract Metadata from Image ---
            rows = ds.get('Rows', 0)
            cols = ds.get('Columns', 0)
            self.lbl_dims.setText(f"{rows} x {cols} x {num_slices}")

            modality = ds.get('Modality', 'N/A')
            self.lbl_modality.setText(str(modality))

            proc_desc = ds.get('RequestedProcedureDescription', 'N/A')
            self.lbl_proc_desc.setText(str(proc_desc))
            # -----------------------------------

            if not hasattr(ds, 'pixel_array'):
                self.lbl_image_preview.setText("No Pixels")
                return

            arr = ds.pixel_array.astype(float)

            # Simple min/max normalization for display
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
            arr = arr.astype(np.uint8)

            # 4. Convert to QImage
            height, width = arr.shape
            bytes_per_line = width
            q_img = QImage(arr.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

            # 5. Set to Label (scaled to fit)
            pixmap = QPixmap.fromImage(q_img)
            self.lbl_image_preview.setPixmap(pixmap.scaled(
                self.lbl_image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

        except Exception as e:
            print(f"Error loading preview: {e}")
            self.lbl_image_preview.setText("Preview Error")

    def populate_table(self):
        """Creates table rows based on the filter config."""
        targets = self.filter_config.get('rois', [])
        self.table.setRowCount(len(targets))

        self.combo_refs = []
        self.current_assignments = []

        options = ["(Unassigned)"] + self.available_rois

        for row, target in enumerate(targets):
            # Column 0: Index ID
            idx_item = QTableWidgetItem(str(target.get('index', 'N/A')))
            idx_item.setFlags(idx_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, idx_item)

            # Column 1: Target Name
            name_item = QTableWidgetItem(target.get('name', 'Unknown'))
            name_item.setFlags(name_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, name_item)

            # Column 2: ComboBox
            combo = QComboBox()
            combo.addItems(options)

            # --- Logic to find best match ---
            best_match = "(Unassigned)"
            include_keys = target.get('include_keys', [])

            for roi in self.available_rois:
                roi_lower = roi.lower()
                if any(k.lower() in roi_lower for k in include_keys):
                    global_exclude = self.filter_config.get('global_exclude', [])
                    if not any(ex.lower() in roi_lower for ex in global_exclude):
                        best_match = roi
                        break

            combo.setCurrentText(best_match)

            # Store initial state
            self.current_assignments.append(best_match)
            self.combo_refs.append(combo)

            combo.currentTextChanged.connect(lambda text, r=row: self.on_combo_changed(r, text))

            self.table.setCellWidget(row, 2, combo)

    def on_combo_changed(self, changed_row, new_text):
        """Handles smart swapping."""
        if new_text == "(Unassigned)":
            self.current_assignments[changed_row] = new_text
            return

        old_text = self.current_assignments[changed_row]

        for other_row, assigned_text in enumerate(self.current_assignments):
            if other_row == changed_row:
                continue

            if assigned_text == new_text:
                # Collision detected: Swap
                self.current_assignments[changed_row] = new_text
                self.current_assignments[other_row] = old_text

                other_combo = self.combo_refs[other_row]
                other_combo.blockSignals(True)
                other_combo.setCurrentText(old_text)
                other_combo.blockSignals(False)
                return

        self.current_assignments[changed_row] = new_text

    def get_roi_assignments(self):
        """
        Extracts the current mappings from the table.

        Returns:
            tuple: (roi_to_index, roi_to_target)
                - roi_to_index: Dict[str, int] -> {'DICOM_ROI_Name': Index_ID}
                - roi_to_target: Dict[str, str] -> {'DICOM_ROI_Name': 'Target_Structure_Name'}
        """
        roi_to_index = {}
        roi_to_target = {}

        for row in range(self.table.rowCount()):
            # 1. Get Selected ROI Name (from ComboBox in Column 2)
            combo = self.table.cellWidget(row, 2)
            selected_roi = combo.currentText()

            # Skip unassigned rows
            if selected_roi == "(Unassigned)":
                continue

            # 2. Get Index ID (from Column 0)
            index_item = self.table.item(row, 0)
            try:
                # Convert to int, or keep as string if you prefer
                index_val = int(index_item.text())
            except ValueError:
                index_val = index_item.text()

            # 3. Get Target Structure Name (from Column 1)
            target_item = self.table.item(row, 1)
            target_name = target_item.text()

            # 4. Populate Dictionaries
            roi_to_index[selected_roi] = index_val
            roi_to_target[selected_roi] = target_name

        return roi_to_index, roi_to_target


class SelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Input Selection")
        self.resize(500, 150)

        # Main vertical layout
        self.main_layout = QVBoxLayout()

        # --- Row 1: DICOM File Selection ---
        self.row1_layout = QHBoxLayout()

        self.lbl_file = QLabel("DICOM File:")
        self.lbl_file.setFixedWidth(80)

        self.txt_file_path = QLineEdit()
        self.txt_file_path.setPlaceholderText("Select a .dcm file...")

        self.btn_browse_file = make_button("coordinate_axes",
                                           "Browse for DICOM RT File",
                                           self.browse_dicom_file)

        self.row1_layout.addWidget(self.lbl_file)
        self.row1_layout.addWidget(self.txt_file_path)
        self.row1_layout.addWidget(self.btn_browse_file)

        # --- Row 2: Directory Selection ---
        self.row2_layout = QHBoxLayout()

        self.lbl_dir = QLabel("Directory:")
        self.lbl_dir.setFixedWidth(80)

        self.txt_dir_path = QLineEdit()
        self.txt_dir_path.setPlaceholderText("Select a folder...")

        self.btn_browse_dir = make_button("folder", "Browse for Output Directory", self.browse_directory)

        self.row2_layout.addWidget(self.lbl_dir)
        self.row2_layout.addWidget(self.txt_dir_path)
        self.row2_layout.addWidget(self.btn_browse_dir)

        # --- Row 3: Next Button ---
        self.row3_layout = QHBoxLayout()
        self.row3_layout.addStretch()

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.open_next_window)
        self.btn_next.setFixedWidth(100)

        self.row3_layout.addWidget(self.btn_next)

        # --- Row 4: Status/Log Box ---
        self.row4_layout = QHBoxLayout()
        self.text_box = QPlainTextEdit()

        self.text_box.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #000000;
                color: #FFFFFF;
                border-radius: 2px;
                border: 1px solid #555555;
            }}
            """
                                    )

        self.text_box.setReadOnly(True)
        self.text_box.setFixedHeight(100)
        self.text_box.setMaximumBlockCount(100)
        self.row4_layout.addWidget(self.text_box)

        # Add layouts to main
        self.main_layout.addLayout(self.row1_layout)
        self.main_layout.addLayout(self.row2_layout)
        self.main_layout.addLayout(self.row4_layout)
        self.main_layout.addLayout(self.row3_layout)

        self.setLayout(self.main_layout)

    def browse_dicom_file(self):
        """Open file dialog filtered for .dcm files"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DICOM File",
            "", # Start directory (empty = current)
            "DICOM Files (*.dcm);;All Files (*)"
        )

        if file_path:
            self.txt_file_path.setText(file_path)

    def browse_directory(self):
        """Open directory selection dialog"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            ""
        )

        if dir_path:
            self.txt_dir_path.setText(dir_path)

    def check_the_dicom(self, dicom_data:DICOMRTStructData):
        self.text_box.clear()
        self.text_box.appendPlainText("Validating DICOM data...")
        try:
            dicom_data["ref_image_series_uid"] = get_ref_image_series_uid(dicom_data['dicom_rtstruct_file'])
            self.text_box.appendPlainText("Reference Image Series UID: " + str(dicom_data["ref_image_series_uid"]) + "\n")

            uids = get_unique_series_uids(dicom_data["dicom_image_set_dir"])
            if dicom_data["ref_image_series_uid"] not in uids:
                self.text_box.appendPlainText("Error: The DICOM RT Reference Image UID is not in the selected DICOM image set directory.")
                return False

            dicom_data["roi_list"] = get_rtstruct_roi_names(dicom_data['dicom_rtstruct_file'])
            self.text_box.appendPlainText(f"Found {len(dicom_data['roi_list'])} ROIs.")

        except (InvalidDicomError, TypeError, OSError) as e:
            self.text_box.appendPlainText(f"Error: The DICOM Data is not valid. {e}")
            return False
        except Exception as e:
            self.text_box.appendPlainText(f"Unexpected Error: {e}")
            return False

        return True

    def open_next_window(self):
        """
        Validates inputs, hides current window, and opens the Processing Window.
        """
        # 1. Validation
        if not self.txt_file_path.text() or not self.txt_dir_path.text():
            self.text_box.appendPlainText("Please select both a DICOM file and an output directory.")
            return

        # 2. Prepare Data
        dicomrt_data = {
            "dicom_rtstruct_file": self.txt_file_path.text(),
            "dicom_image_set_dir": self.txt_dir_path.text(),
            "dicom_ref_image_series_uid": None,
            "roi_list": [],
            "struct_mask": np.array([]),
            "image": np.array([]),
            "ref_image_series_uid": None
        }

        # 3. Check Data
        if not self.check_the_dicom(dicomrt_data):
            return

        # 4. Navigate
        self.hide()
        self.next_window = ProcessingWindow(dicomrt_data)
        result = self.next_window.exec()

        if result == QDialog.DialogCode.Rejected:
            self.show()
        elif result == QDialog.DialogCode.Accepted:
            self.processed_data = self.next_window.processed_data
            self.accept()

    def resizeEvent(self, event):
        if hasattr(self, 'btn_next'):
            new_width = int(self.width() * 0.25)
            self.btn_next.setFixedWidth(new_width)
        super().resizeEvent(event)

def setup_default_logging(log_file='app.log'):
    """
    Sets up a robust logging configuration that logs all messages (DEBUG and above)
    to both a file and the console.
    """
    # 1. Define the Logger (root logger)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Sets the lowest severity level to handle

    # Ensure no handlers from previous configurations persist
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # 2. Define the Formatter
    # The format includes: Timestamp, Log Level, Logger Name, File/Line, Message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - (%(filename)s:%(lineno)d) - %(message)s'
    )

    # --- Setup Handlers ---

    # 3. File Handler: Logs all messages (DEBUG and above) to a file
    # 'a' mode means append
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 4. Stream Handler: Logs all messages (DEBUG and above) to the console
    # Use sys.stdout for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    print(f"Logging configured. Messages DEBUG and above will be written to the console and '{os.path.abspath(log_file)}'.")

if __name__ == "__main__":
    setup_default_logging()
    copy_custom_ui_icons()
    app = QApplication(sys.argv)
    NAPARI_AVAILABLE=False
    if NAPARI_AVAILABLE:
        customize_stylesheet(app)

    window = SelectionDialog()
    window.show()

    sys.exit(app.exec())