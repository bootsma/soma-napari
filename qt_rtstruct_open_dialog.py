import sys
import numpy as np
from typing import TypedDict

from PyQt6.QtGui import QPalette, QColor

from mira_core.dicom_utils import get_ref_image_series_uid, get_unique_series_uids, get_rtstruct_roi_names
from pydicom.errors import InvalidDicomError
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QWidget, QTextEdit, QPlainTextEdit
)

from qt_theme_utils import (
    copy_custom_ui_icons, customize_stylesheet,
    make_button, NAPARI_AVAILABLE
)

# --- Napari Theme Imports ---
# Importing these modules triggers the build of built-in themes
# and registers the 'theme_dark:/' Qt resource paths.
"""
try:
    from napari.utils.theme import get_theme
    from napari._qt.qt_resources import get_stylesheet
    from napari._qt.widgets.qt_viewer_buttons import (
        QtViewerPushButton
    )
    from napari.utils.translations import trans
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    print("Napari not found. Using default Qt theme.")
"""

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
            'include_keys':[
                'bladder'
            ],
            'exclude_keys':[
            ]
        },
        {
            'name':'Rectum',
            'index':2,
            'include_keys':[
                'rectum'
            ]
        },
        {
            'name':'Sigmoid',
            'index':4,
            'include_keys':[
                'sigmoid'
            ]
        },
        {
            'name':'SmallBowel',
            'index':3,
            'include_keys':[
                'smallbowel',
                'bowel'
            ]
        },

    ],
    'global_exclude':[
        'fake',
        'inside'
    ]

}


fasdf

class ProcessingWindow(QDialog):
    """
    Placeholder for the window that opens after 'Next' is clicked.
    We will write the detailed code for this later.
    """
    def __init__(self, dicom_data:DICOMRTStructData, filter:dict=None):
        super().__init__()
        if not filter:
            filter = DEFAULT_ROI_FILTER



        self.setWindowTitle("Processing Window")
        self.resize(400, 300)

        layout = QVBoxLayout()
        label = QLabel("This is the next window.\n(Logic to be implemented later)")
        layout.addWidget(label)
        self.setLayout(layout)


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
        self.lbl_file.setFixedWidth(80) # Fixed width for alignment

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
        self.row3_layout.addStretch() # Pushes button to the right

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.open_next_window)
        self.btn_next.setFixedWidth(100)

        self.row3_layout.addWidget(self.btn_next)


        self.row4_layout = QHBoxLayout()
        self.text_box = QPlainTextEdit()
        self.text_box.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #000000;
                color: #FFFFFF;
                border-radius: 2px;
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
        #open the
        self.text_box.clear()
        self.text_box.appendPlainText("Validating DICOM data...")
        try:

            dicom_data["ref_image_series_uid"] = get_ref_image_series_uid(dicom_data['dicom_rtstruct_file'])
            self.text_box.appendPlainText("Reference Image Series UID: " + dicom_data["ref_image_series_uid"] + "\n")
            uids = get_unique_series_uids(dicom_data["dicom_image_set_dir"])
            if dicom_data["ref_image_series_uid"] not in uids:
                self.text_box.appendPlainText("The DICOM RT Reference Image UID is not in the DICOM image set.")
                return False
            dicom_data["roi_list"] = get_rtstruct_roi_names(dicom_data['dicom_rtstruct_file'])
            self.text_box.appendPlainText("ROI List: " + str(dicom_data["roi_list"]) + "\n")

        except (InvalidDicomError, TypeError, OSError):
            self.text_box.appendPlainText(f"The DICOM Data is not valid {e}.")
            return False



        return True


    def open_next_window(self):
        """Closes this dialog and opens the next one"""
        # Optional: Add validation here to ensure paths are selected
        if not self.txt_file_path.text() or not self.txt_dir_path.text():
            self.text_box.appendPlainText("Please select both a DICOM file and an output directory.")
            return

        # --- Get the DICOM data ---
        dicomrt_data = {
            "dicom_rtstruct_file": self.txt_file_path.text(),
            "dicom_image_set_dir": self.txt_dir_path.text(),
            "dicom_ref_image_series_uid":None,
            "roi_list": [],
            "struct_mask": np.array([]),
            "image": np.array([]),
            "ref_image_series_uid":None
        }

        self.check_the_dicom(dicomrt_data)
        dbg=True
        if dbg:
            return
        # Close/Hide current window
        self.accept() # Standard way to close a dialog successfully

        # Open the next window
        self.next_window = ProcessingWindow()
        self.next_window.exec()

if __name__ == "__main__":
    copy_custom_ui_icons()
    app = QApplication(sys.argv)

    # --- Apply Napari Theme (Standalone) ---
    if NAPARI_AVAILABLE:
        # get_theme('dark') ensures the 'dark' theme resources are built
        # and search paths (theme_dark:/) are registered with Qt.
        customize_stylesheet(app)


    window = SelectionDialog()
    window.show()

    sys.exit(app.exec())