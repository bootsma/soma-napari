import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QWidget
)

from qt_theme_utils import copy_custom_ui_icons, customize_stylesheet


class ProcessingWindow(QDialog):
    """
    Placeholder for the window that opens after 'Next' is clicked.
    We will write the detailed code for this later.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing Window")
        self.resize(400, 300)

        layout = QVBoxLayout()
        label = QLabel("This is the next window.\n(Logic to be implemented later)")
        layout.addWidget(label)
        self.setLayout(layout)


class DICOMRTSelectionDialog(QDialog):
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

        self.btn_browse_file = QPushButton("Browse")
        self.btn_browse_file.clicked.connect(self.browse_dicom_file)

        self.row1_layout.addWidget(self.lbl_file)
        self.row1_layout.addWidget(self.txt_file_path)
        self.row1_layout.addWidget(self.btn_browse_file)

        # --- Row 2: Directory Selection ---
        self.row2_layout = QHBoxLayout()

        self.lbl_dir = QLabel("Directory:")
        self.lbl_dir.setFixedWidth(80)

        self.txt_dir_path = QLineEdit()
        self.txt_dir_path.setPlaceholderText("Select a folder...")

        self.btn_browse_dir = QPushButton("Browse")
        self.btn_browse_dir.clicked.connect(self.browse_directory)

        self.row2_layout.addWidget(self.lbl_dir)
        self.row2_layout.addWidget(self.txt_dir_path)
        self.row2_layout.addWidget(self.btn_browse_dir)

        # --- Row 3: Next Button ---
        self.row3_layout = QHBoxLayout()
        self.row3_layout.addStretch() # Pushes button to the right

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.open_next_window)

        self.row3_layout.addWidget(self.btn_next)

        # Add layouts to main
        self.main_layout.addLayout(self.row1_layout)
        self.main_layout.addLayout(self.row2_layout)
        self.main_layout.addStretch() # Adds space between inputs and Next button
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

    def open_next_window(self):
        """Closes this dialog and opens the next one"""
        # Optional: Add validation here to ensure paths are selected
        # if not self.txt_file_path.text() or not self.txt_dir_path.text():
        #     return

        # Close/Hide current window
        self.accept() # Standard way to close a dialog successfully

        # Open the next window
        self.next_window = ProcessingWindow()
        self.next_window.exec()

if __name__ == "__main__":
    copy_custom_ui_icons()
    app = QApplication.instance() or QApplication(sys.argv)

    window = DICOMRTSelectionDialog()

    customize_stylesheet(app)
    window.show()

    sys.exit(app.exec())