import os
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton, QComboBox,
    QLineEdit, QFormLayout, QHBoxLayout, QMessageBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QScrollArea, QDoubleSpinBox,
    QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

from orca.inference import InferenceEngine
from orca.logger import logger


class InferenceWorkerThread(QThread):
    """Background thread for running inference."""
    finished = Signal(bool, str, dict)  # success, message, results
    
    def __init__(self, engine: InferenceEngine, inputs: dict):
        super().__init__()
        self.engine = engine
        self.inputs = inputs
    
    def run(self):
        try:
            results = self.engine.infer(self.inputs)
            self.finished.emit(True, "Inference completed successfully!", results)
        except Exception as e:
            self.finished.emit(False, f"Inference failed: {str(e)}", {})


class InferenceTab(QWidget):
    """Standalone inference tab widget."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.inference_engine = None
        self.inference_worker = None
        self.inference_input_widgets = {}
        self._build_ui()
        self._refresh_checkpoints()

    def _build_ui(self):
        layout = QVBoxLayout()
        
        # ============= Model Selection Section =============
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        
        checkpoint_label = QLabel("Checkpoint:")
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.currentTextChanged.connect(self._on_checkpoint_selected)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_checkpoints)
        checkpoint_layout.addWidget(self.checkpoint_combo, 1)
        checkpoint_layout.addWidget(refresh_btn)
        model_layout.addRow(checkpoint_label, checkpoint_layout)
        
        self.model_info_label = QLabel("No model loaded")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("font-size: 10px; color: #666;")
        model_layout.addRow("Model Info:", self.model_info_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # ============= Input Parameters Section =============
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        
        input_scroll = QScrollArea()
        input_scroll.setWidgetResizable(True)
        input_scroll_widget = QWidget()
        self.input_form_layout = QFormLayout()
        self.input_form_layout.setHorizontalSpacing(10)
        self.input_form_layout.setVerticalSpacing(4)
        self.input_form_layout.setContentsMargins(6, 6, 6, 6)
        input_scroll_widget.setLayout(self.input_form_layout)
        input_scroll.setWidget(input_scroll_widget)
        input_layout.addWidget(input_scroll)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # ============= Inference Button =============
        button_layout = QHBoxLayout()
        self.infer_button = QPushButton("Run Inference")
        self.infer_button.setMinimumHeight(40)
        self.infer_button.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.infer_button.clicked.connect(self._run_inference)
        self.infer_button.setEnabled(False)
        button_layout.addWidget(self.infer_button)
        layout.addLayout(button_layout)
        
        # ============= Progress Display =============
        self.inference_progress = QProgressBar()
        self.inference_progress.setVisible(False)
        layout.addWidget(self.inference_progress)
        
        # ============= Results Section =============
        results_group = QGroupBox("Inference Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Output Name", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.inference_status = QLabel("Ready")
        self.inference_status.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self.inference_status)
        
        self.setLayout(layout)

    def _refresh_checkpoints(self):
        """Refresh list of available checkpoints."""
        try:
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            engine = InferenceEngine(checkpoint_dir)
            checkpoints = engine.list_checkpoints()

            current_selection = self.checkpoint_combo.currentText()
            self.checkpoint_combo.clear()

            if checkpoints:
                self.checkpoint_combo.addItems(checkpoints)
                index = self.checkpoint_combo.findText(current_selection)
                if index >= 0:
                    self.checkpoint_combo.setCurrentIndex(index)
            else:
                self.checkpoint_combo.addItem("No checkpoints found")
                self.model_info_label.setText(
                    f"No ONNX files found in {checkpoint_dir}\n\n"
                    "Please add .onnx files to the checkpoints directory."
                )
                self.infer_button.setEnabled(False)
        except Exception as e:
            logger.error(f"Error refreshing checkpoints: {e}")
            self.model_info_label.setText(f"Error loading checkpoints: {str(e)}")
            self.infer_button.setEnabled(False)

    def _on_checkpoint_selected(self):
        """Handle checkpoint selection change."""
        checkpoint_name = self.checkpoint_combo.currentText()

        if not checkpoint_name or checkpoint_name == "No checkpoints found":
            self.infer_button.setEnabled(False)
            self.model_info_label.setText("No model selected")
            return

        try:
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            self.inference_engine = InferenceEngine(checkpoint_dir)
            self.inference_engine.load_checkpoint(checkpoint_name)

            model_info = self.inference_engine.get_model_info()
            info_text = f"Path: {model_info['path']}\n\n"

            inputs = self.inference_engine.get_input_specs()
            outputs = self.inference_engine.get_output_specs()
            self.model_info_label.setText(info_text)

            self._populate_input_fields(inputs)

            self.infer_button.setEnabled(True)
            self.inference_status.setText("Ready - Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            self.model_info_label.setText(f"Error: {str(e)}")
            self.infer_button.setEnabled(False)
            self.inference_status.setText(f"Error: {str(e)}")

    def _populate_input_fields(self, input_specs: dict):
        """Populate input parameter form fields based on model inputs."""
        while self.input_form_layout.rowCount() > 0:
            self.input_form_layout.removeRow(0)
        self.inference_input_widgets = {}

        for input_name, spec in input_specs.items():
            shape = spec.get('shape', [])
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1e6, 1e6)
            spinbox.setValue(0.0)
            spinbox.setDecimals(4)
            spinbox.setMaximumWidth(110)
            spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spinbox.setAlignment(Qt.AlignRight)

            self.inference_input_widgets[input_name] = {
                'widget': spinbox,
                'type': 'scalar',
                'spinbox': spinbox,
                'shape': shape
            }

            self.input_form_layout.addRow(f"{input_name}:", spinbox)

    def _run_inference(self):
        """Run inference with current input parameters."""
        if not self.inference_engine or not self.inference_engine.is_model_loaded():
            QMessageBox.warning(self, "Error", "No model loaded")
            return

        try:
            inputs = {}

            for input_name, widget_info in self.inference_input_widgets.items():
                widget_type = widget_info['type']

                if widget_type == 'scalar':
                    value = widget_info['spinbox'].value()
                    shape = widget_info.get('shape') or []
                    if shape:
                        normalized_shape = []
                        for dim in shape:
                            if isinstance(dim, int) and dim > 0:
                                normalized_shape.append(dim)
                            else:
                                normalized_shape.append(1)
                        inputs[input_name] = np.full(normalized_shape, value, dtype=np.float32)
                    else:
                        inputs[input_name] = np.array([value], dtype=np.float32)

                elif widget_type == '1d_array':
                    spinboxes = widget_info['spinboxes']
                    total_elements = widget_info['total_elements']
                    values = [sb.value() for sb in spinboxes]

                    while len(values) < total_elements:
                        values.append(0.0)

                    inputs[input_name] = np.array(values[:total_elements], dtype=np.float32).reshape(-1)

                elif widget_type == '2d_array':
                    table = widget_info['table']
                    rows, cols = widget_info['total_shape']
                    values = []

                    for i in range(rows):
                        row = []
                        for j in range(cols):
                            widget = table.cellWidget(min(i, table.rowCount()-1), min(j, table.columnCount()-1))
                            if widget and isinstance(widget, QDoubleSpinBox):
                                row.append(widget.value())
                            else:
                                row.append(0.0)
                        values.append(row)

                    inputs[input_name] = np.array(values, dtype=np.float32)

            self.infer_button.setEnabled(False)
            self.inference_progress.setRange(0, 0)
            self.inference_progress.setVisible(True)
            self.inference_status.setText("Running inference...")

            self.inference_worker = InferenceWorkerThread(self.inference_engine, inputs)
            self.inference_worker.finished.connect(self._on_inference_finished)
            self.inference_worker.start()

        except Exception as e:
            logger.error(f"Error preparing inference: {e}")
            QMessageBox.critical(self, "Error", f"Error preparing inference:\n{str(e)}")
            self.infer_button.setEnabled(True)
            self.inference_progress.setVisible(False)

    @Slot(bool, str, dict)
    def _on_inference_finished(self, success: bool, message: str, results: dict):
        """Handle inference completion."""
        self.infer_button.setEnabled(True)
        self.inference_progress.setVisible(False)

        if success:
            self.results_table.setRowCount(0)

            for i, (output_name, output_value) in enumerate(results.items()):
                if not isinstance(output_value, np.ndarray):
                    output_value = np.array(output_value)

                flat_value = output_value.flatten()

                row = i
                self.results_table.insertRow(row)

                name_item = QTableWidgetItem(output_name)
                self.results_table.setItem(row, 0, name_item)

                if len(flat_value) == 1:
                    value_text = f"{flat_value[0]:.6f}"
                else:
                    value_text = f"Shape: {output_value.shape}\n"
                    value_text += "[" + ", ".join(f"{v:.6f}" for v in flat_value[:10])
                    if len(flat_value) > 10:
                        value_text += f", ... ({len(flat_value)} total)]"
                    else:
                        value_text += "]"

                value_item = QTableWidgetItem(value_text)
                self.results_table.setItem(row, 1, value_item)

            self.inference_status.setText("✓ Inference completed successfully")
            QMessageBox.information(self, "Success", message)
        else:
            self.inference_status.setText(f"✗ {message}")
            QMessageBox.critical(self, "Error", message)


def create_inference_tab(parent: QWidget | None = None) -> InferenceTab:
    """Factory retained for backward compatibility."""
    return InferenceTab(parent)