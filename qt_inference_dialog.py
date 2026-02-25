import logging
import numpy as np
import SimpleITK as sitk
import sys
import os
import json

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QComboBox, QProgressBar, QMessageBox
)
from napari.layers import Image

# Adjusting path to import hydra and mira_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hydra")))
try:
    from hydra.inference_servers.exe.inference_client import InferenceClient
except ImportError:
    # Fallback if hydra is not in the expected relative path
    try:
        from inference_client import InferenceClient
    except ImportError:
        print("Error: Could not import InferenceClient from hydra.")
        InferenceClient = None

try:
    from mira_core.volume_info import VolumeInformation
except ImportError:
    print("Error: Could not import VolumeInformation from mira_core.")
    VolumeInformation = None



class InferenceWorker(QObject):
    """
    Handles ZMQ communication with the inference server in a separate thread.
    """
    finished = pyqtSignal(object)  # Returns sitk.Image
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, client: InferenceClient, model_name: str, input_image: sitk.Image):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.input_image = input_image

    def run(self):
        try:
            # Note: infer_image is a synchronous call that blocks.
            # We run it in this thread to keep the UI responsive.
            result_image = self.client.infer_image(
                model_name=self.model_name,
                input_image=self.input_image,
                send_as_path=False
            )
            self.finished.emit(result_image)
        except Exception as e:
            self.error.emit(str(e))


class InferenceDialog(QDialog):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.client = None
        self.result_image = None
        
        # Load configuration
        self.config_path = os.path.join(os.path.dirname(__file__), "config", "main_config.json")
        self.config = self.load_config()

        self.setWindowTitle("AI Inference Client")
        self.resize(450, 300)
        self.setup_ui()

    def load_config(self):
        """Loads configuration from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # Default config if file doesn't exist or fails to load
        return {
            "inference_server": {
                "host": "localhost",
                "port": "5789"
            }
        }

    def save_config(self):
        """Saves the current server configuration to JSON file."""
        try:
            # Ensure the config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            self.config["inference_server"] = {
                "host": self.txt_host.text(),
                "port": self.txt_port.text()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def setup_ui(self):
        layout = QVBoxLayout()

        # --- Connection Group ---
        conn_group = QGroupBox("Server Connection")
        conn_layout = QFormLayout()

        server_cfg = self.config.get("inference_server", {})
        self.txt_host = QLineEdit(server_cfg.get("host", "localhost"))
        self.txt_port = QLineEdit(server_cfg.get("port", "5789"))
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.on_connect)

        conn_layout.addRow("Host:", self.txt_host)
        conn_layout.addRow("Port:", self.txt_port)
        conn_layout.addRow("", self.btn_connect)
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # --- Inference Options Group ---
        opts_group = QGroupBox("Inference Options")
        opts_layout = QFormLayout()

        self.combo_model = QComboBox()
        self.combo_model.setEnabled(False)
        
        self.combo_volume = QComboBox()
        self.populate_volumes()

        opts_layout.addRow("Model:", self.combo_model)
        opts_layout.addRow("Target Volume:", self.combo_volume)
        opts_group.setLayout(opts_layout)
        layout.addWidget(opts_group)

        # --- Execution Group ---
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        layout.addWidget(self.pbar)

        self.btn_run = QPushButton("Run Inference")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.on_run)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def populate_volumes(self):
        """Finds all Image layers in Napari."""
        self.combo_volume.clear()
        image_layers = [L.name for L in self.viewer.layers if isinstance(L, Image)]
        self.combo_volume.addItems(image_layers)
        
        # Select active layer if it's an image
        active = self.viewer.layers.selection.active
        if active and active.name in image_layers:
            self.combo_volume.setCurrentText(active.name)

    def on_connect(self):
        host = self.txt_host.text()
        port = self.txt_port.text()
        
        try:
            if self.client:
                self.client.close()
            
            self.client = InferenceClient(host=host, port=port)
            models = self.client.get_models()
            
            if not models:
                QMessageBox.warning(self, "Connection", "Connected, but no models found on server.")
                return

            self.combo_model.clear()
            self.combo_model.addItems(models)
            self.combo_model.setEnabled(True)
            self.btn_run.setEnabled(True)
            
            # Save successful connection settings
            self.save_config()
            
            QMessageBox.information(self, "Connection", f"Successfully connected to {host}:{port}")

        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect to server: {e}")

    def on_run(self):
        if not self.combo_volume.currentText():
            QMessageBox.warning(self, "Inference", "Please select a volume.")
            return

        model_name = self.combo_model.currentText()
        layer_name = self.combo_volume.currentText()
        layer = self.viewer.layers[layer_name]

        # 1. Convert Napari Image Layer to sitk.Image
        try:
            # Get data as numpy array
            data = layer.data
            if hasattr(data, 'compute'): # Handle dask arrays
                data = data.compute()

            # Create sitk.Image
            # SimpleITK is XYZ, Napari is ZYX
            sitk_img = sitk.GetImageFromArray(data)
            
            # Use layer.scale and layer.translate if available
            # Napari: (sz, sy, sx), SITK: (sx, sy, sz)
            scale = getattr(layer, 'scale', (1.0, 1.0, 1.0))
            translate = getattr(layer, 'translate', (0.0, 0.0, 0.0))
            
            sitk_img.SetSpacing(np.flip(scale).tolist())
            sitk_img.SetOrigin(np.flip(translate).tolist())
            
            # If we have volume_info in metadata, we can use it for direction
            if 'volume_info' in layer.metadata:
                vol_info = layer.metadata['volume_info']
                sitk_img.SetDirection(vol_info.direction)

        except Exception as e:
            QMessageBox.critical(self, "Data Conversion Error", f"Failed to convert layer to SimpleITK: {e}")
            return

        # 2. Start Worker Thread
        self.pbar.setVisible(True)
        self.pbar.setRange(0, 0) # Indeterminate
        self.btn_run.setEnabled(False)
        self.btn_connect.setEnabled(False)

        self.thread = QThread()
        self.worker = InferenceWorker(self.client, model_name, sitk_img)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        
        self.thread.start()

    def on_worker_finished(self, result_image):
        self.result_image = result_image
        self.thread.quit()
        self.thread.wait()
        self.accept()

    def on_worker_error(self, error_msg):
        self.thread.quit()
        self.thread.wait()
        self.pbar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_connect.setEnabled(True)
        QMessageBox.critical(self, "Inference Error", f"Server error during inference: {error_msg}")

    def closeEvent(self, event):
        if self.client:
            try:
                self.client.close()
            except:
                pass
        super().closeEvent(event)
