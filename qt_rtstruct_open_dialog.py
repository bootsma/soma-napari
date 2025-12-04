import sys
import os
import numpy as np
import pydicom
from typing import TypedDict, List, Dict

from PyQt6.QtGui import QPalette, QColor, QPixmap, QImage
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QFileDialog, QWidget, QTextEdit,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QMessageBox
)

from mira_core.dicom_utils import get_ref_image_series_uid, get_unique_series_uids, get_rtstruct_roi_names
from pydicom.errors import InvalidDicomError

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
    struct_mask: np.ndarray
    image: np.ndarray

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

class ProcessingWindow(QDialog):
    """
    Step 2: ROI Mapping and Metadata Verification.
    """
    def __init__(self, dicom_data: DICOMRTStructData, filter_config: dict = None):
        super().__init__()
        self.setWindowTitle("ROI Assignment")
        self.resize(900, 600) # Slightly wider for image preview

        self.dicom_data = dicom_data
        self.filter_config = filter_config or DEFAULT_ROI_FILTER
        self.available_rois = sorted(dicom_data.get('roi_list', []))
        self.combo_refs: List[QComboBox] = []
        self.current_assignments: List[str] = [] # Tracks text in each row to enable swapping

        self.setup_ui()
        self.load_patient_data()
        self.load_preview_image()
        self.populate_table()

    def setup_ui(self):
        layout = QVBoxLayout()

        # --- 1. Patient Information Group ---
        self.info_group = QGroupBox("Patient Information")
        # Main layout for group is Horizontal: Text on Left, Image on Right
        group_layout = QHBoxLayout()

        # Left Side: Text Info
        self.info_layout = QFormLayout()
        self.lbl_name = QLabel("Loading...")
        self.lbl_id = QLabel("Loading...")
        self.lbl_date = QLabel("Loading...")

        self.info_layout.addRow("Patient Name:", self.lbl_name)
        self.info_layout.addRow("Patient ID:", self.lbl_id)
        self.info_layout.addRow("Study Date:", self.lbl_date)

        # Right Side: Image Preview
        self.lbl_image_preview = QLabel("No Preview")
        self.lbl_image_preview.setFixedSize(150, 150)
        self.lbl_image_preview.setStyleSheet("border: 1px solid #555; background-color: #000;")
        self.lbl_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add to group layout
        group_layout.addLayout(self.info_layout, stretch=1)
        group_layout.addWidget(self.lbl_image_preview)

        self.info_group.setLayout(group_layout)
        layout.addWidget(self.info_group)

        # --- 2. ROI Mapping Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(3) # Increased to 3 columns
        self.table.setHorizontalHeaderLabels(["ID", "Target Structure", "DICOM Contour"])

        # Header resizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Target
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)          # Combo

        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self.table)

        # --- 3. Action Buttons ---
        btn_layout = QHBoxLayout()

        self.btn_back = QPushButton("Back")
        self.btn_back.setFixedWidth(100)
        self.btn_back.clicked.connect(self.reject)

        self.btn_process = QPushButton("Process")
        self.btn_process.setFixedWidth(100)
        self.btn_process.clicked.connect(self.accept)

        btn_layout.addWidget(self.btn_back)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_process)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

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
        """Loads a middle slice from the DICOM image directory and displays it."""
        image_dir = self.dicom_data.get('dicom_image_set_dir')
        if not image_dir or not os.path.exists(image_dir):
            return

        try:
            # 1. Find DICOM files
            files = [f for f in os.listdir(image_dir) if f.lower().endswith('.dcm')]
            if not files:
                self.lbl_image_preview.setText("No DICOMs")
                return

            # 2. Pick middle slice (naive sort by filename usually works for simple cases)
            files.sort()
            mid_idx = len(files) // 2
            dicom_path = os.path.join(image_dir, files[mid_idx])

            # 3. Read and Normalize
            ds = pydicom.dcmread(dicom_path)
            if not hasattr(ds, 'pixel_array'):
                self.lbl_image_preview.setText("No Pixels")
                return

            arr = ds.pixel_array.astype(float)

            # Simple min/max normalization for display
            # (Note: For medical accuracy, window/level logic is preferred,
            # but min/max is sufficient for a "thumbnail" preview)
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
            arr = arr.astype(np.uint8)

            # 4. Convert to QImage
            height, width = arr.shape
            bytes_per_line = width
            # Format_Grayscale8 is ideal for medical images
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
            self.accept()

    def resizeEvent(self, event):
        if hasattr(self, 'btn_next'):
            new_width = int(self.width() * 0.25)
            self.btn_next.setFixedWidth(new_width)
        super().resizeEvent(event)

if __name__ == "__main__":
    copy_custom_ui_icons()
    app = QApplication(sys.argv)

    if NAPARI_AVAILABLE:
        customize_stylesheet(app)

    window = SelectionDialog()
    window.show()

    sys.exit(app.exec())