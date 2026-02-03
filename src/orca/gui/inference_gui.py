import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QDoubleSpinBox,
    QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from orca.inference import InferenceEngine
from orca.logger import logger


class InferenceWorkerThread(QThread):
    """Background thread for running inference."""

    finished = Signal(
        bool, str, dict, np.ndarray
    )  # success, message, results, frequency_array

    def __init__(
        self,
        engine: InferenceEngine,
        inputs_list: list,
        frequency_points: np.ndarray | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.inputs_list = inputs_list  # List of input dicts for multiple inferences
        self.frequency_points = (
            frequency_points if frequency_points is not None else np.array([])
        )  # Array of frequency points

    def run(self):
        try:
            all_results = {}

            # Run inference for each input set
            for inputs in self.inputs_list:
                results = self.engine.infer(inputs)

                # Combine results from all inferences
                for output_name, output_value in results.items():
                    if output_name not in all_results:
                        all_results[output_name] = []
                    all_results[output_name].append(output_value)

            # Convert lists to arrays
            for output_name in all_results:
                all_results[output_name] = np.array(all_results[output_name]).flatten()

            self.finished.emit(
                True,
                "Inference completed successfully!",
                all_results,
                self.frequency_points,
            )
        except Exception as e:
            self.finished.emit(False, f"Inference failed: {str(e)}", {}, np.array([]))


class InferenceTab(QWidget):
    """Standalone inference tab widget."""

    def __init__(
        self, geometry_registry: dict | None = None, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.geometry_registry = geometry_registry or {}
        self.inference_engine = None
        self.inference_worker = None
        self.inference_input_widgets = {}
        self.active_geometry = None
        self.active_geometry_class = None
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

    def _refresh_checkpoints(self, preferred_checkpoint: str | None = None):
        """Refresh list of available checkpoints."""
        try:
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            engine = InferenceEngine(checkpoint_dir)
            checkpoints = engine.list_checkpoints()

            current_selection = self.checkpoint_combo.currentText()
            self.checkpoint_combo.clear()

            if checkpoints:
                self.checkpoint_combo.addItems(checkpoints)
                target = (
                    preferred_checkpoint or self.active_geometry or current_selection
                )
                if target:
                    index = self.checkpoint_combo.findText(target)
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

    def set_active_geometry(self, geometry_name: str):
        """Update active geometry and prefer matching checkpoint."""
        self.active_geometry = geometry_name
        geometry_info = self.geometry_registry.get(geometry_name)
        self.active_geometry_class = geometry_info["class"] if geometry_info else None
        self._refresh_checkpoints(preferred_checkpoint=geometry_name + ".onnx")

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
            shape = spec.get("shape", [])

            # Special handling for frequency parameter
            if "frequency" in input_name.lower():
                # Create three spinboxes for fstart, fstop, fstep
                freq_layout = QHBoxLayout()
                freq_spinboxes = {}

                for freq_param in ["fstart [GHz]", "fstop [GHz]", "fstep [GHz]"]:
                    spinbox = QDoubleSpinBox()
                    spinbox.setRange(1.0, 200.0)
                    if freq_param == "fstart [GHz]":
                        spinbox.setValue(1.0)
                    elif freq_param == "fstop [GHz]":
                        spinbox.setValue(200.0)
                    else:
                        spinbox.setValue(1.0)
                    spinbox.setDecimals(4)
                    spinbox.setMaximumWidth(110)
                    spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                    spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
                    freq_spinboxes[freq_param] = spinbox
                    freq_layout.addWidget(QLabel(f"{freq_param}:"))
                    freq_layout.addWidget(spinbox)

                freq_widget = QWidget()
                freq_widget.setLayout(freq_layout)

                self.inference_input_widgets[input_name] = {
                    "widget": freq_widget,
                    "type": "frequency_range",
                    "spinboxes": freq_spinboxes,
                    "shape": shape,
                }

                self.input_form_layout.addRow(f"{input_name}:", freq_widget)
            else:
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-1e6, 1e6)
                spinbox.setValue(0.0)
                spinbox.setDecimals(4)
                spinbox.setMaximumWidth(110)
                spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)

                self.inference_input_widgets[input_name] = {
                    "widget": spinbox,
                    "type": "scalar",
                    "spinbox": spinbox,
                    "shape": shape,
                }

                self.input_form_layout.addRow(f"{input_name}:", spinbox)

    def _run_inference(self):
        """Run inference with current input parameters."""
        if not self.inference_engine or not self.inference_engine.is_model_loaded():
            QMessageBox.warning(self, "Error", "No model loaded")
            return

        try:
            # Check if frequency parameter exists
            has_frequency = any(
                "frequency" in name.lower()
                for name in self.inference_input_widgets.keys()
            )

            if has_frequency:
                # Get frequency range parameters
                freq_param_name = next(
                    name
                    for name in self.inference_input_widgets.keys()
                    if "frequency" in name.lower()
                )
                freq_spinboxes = self.inference_input_widgets[freq_param_name][
                    "spinboxes"
                ]
                fstart = freq_spinboxes["fstart [GHz]"].value()
                fstop = freq_spinboxes["fstop [GHz]"].value()
                fstep = freq_spinboxes["fstep [GHz]"].value()

                # Generate frequency points
                frequency_points = np.arange(fstart, fstop + fstep, fstep)

                # Prepare list of inputs for each frequency point
                inputs_list = []
                for freq in frequency_points:
                    inputs = {}

                    for input_name, widget_info in self.inference_input_widgets.items():
                        widget_type = widget_info["type"]

                        if "frequency" in input_name.lower():
                            # Use current frequency value
                            shape = widget_info.get("shape") or []

                            # Multiply value by 1e9 to convert GHz to Hz
                            freq = freq * 1e9
                            if shape:
                                normalized_shape = []
                                for dim in shape:
                                    if isinstance(dim, int) and dim > 0:
                                        normalized_shape.append(dim)
                                    else:
                                        normalized_shape.append(1)
                                inputs[input_name] = np.full(
                                    normalized_shape, freq, dtype=np.float32
                                )
                            else:
                                inputs[input_name] = np.array([freq], dtype=np.float32)

                        elif widget_type == "scalar":
                            value = widget_info["spinbox"].value()
                            shape = widget_info.get("shape") or []
                            if shape:
                                normalized_shape = []
                                for dim in shape:
                                    if isinstance(dim, int) and dim > 0:
                                        normalized_shape.append(dim)
                                    else:
                                        normalized_shape.append(1)
                                inputs[input_name] = np.full(
                                    normalized_shape, value, dtype=np.float32
                                )
                            else:
                                inputs[input_name] = np.array([value], dtype=np.float32)

                        elif widget_type == "1d_array":
                            spinboxes = widget_info["spinboxes"]
                            total_elements = widget_info["total_elements"]
                            values = [sb.value() for sb in spinboxes]

                            while len(values) < total_elements:
                                values.append(0.0)

                            inputs[input_name] = np.array(
                                values[:total_elements], dtype=np.float32
                            ).reshape(-1)

                        elif widget_type == "2d_array":
                            table = widget_info["table"]
                            rows, cols = widget_info["total_shape"]
                            values = []

                            for i in range(rows):
                                row = []
                                for j in range(cols):
                                    widget = table.cellWidget(
                                        min(i, table.rowCount() - 1),
                                        min(j, table.columnCount() - 1),
                                    )
                                    if widget and isinstance(widget, QDoubleSpinBox):
                                        row.append(widget.value())
                                    else:
                                        row.append(0.0)
                                values.append(row)

                            inputs[input_name] = np.array(values, dtype=np.float32)

                    inputs_list.append(inputs)
            else:
                # Single inference without frequency sweep
                inputs = {}

                for input_name, widget_info in self.inference_input_widgets.items():
                    widget_type = widget_info["type"]

                    if widget_type == "scalar":
                        value = widget_info["spinbox"].value()
                        shape = widget_info.get("shape") or []
                        if shape:
                            normalized_shape = []
                            for dim in shape:
                                if isinstance(dim, int) and dim > 0:
                                    normalized_shape.append(dim)
                                else:
                                    normalized_shape.append(1)
                            inputs[input_name] = np.full(
                                normalized_shape, value, dtype=np.float32
                            )
                        else:
                            inputs[input_name] = np.array([value], dtype=np.float32)

                    elif widget_type == "1d_array":
                        spinboxes = widget_info["spinboxes"]
                        total_elements = widget_info["total_elements"]
                        values = [sb.value() for sb in spinboxes]

                        while len(values) < total_elements:
                            values.append(0.0)

                        inputs[input_name] = np.array(
                            values[:total_elements], dtype=np.float32
                        ).reshape(-1)

                    elif widget_type == "2d_array":
                        table = widget_info["table"]
                        rows, cols = widget_info["total_shape"]
                        values = []

                        for i in range(rows):
                            row = []
                            for j in range(cols):
                                widget = table.cellWidget(
                                    min(i, table.rowCount() - 1),
                                    min(j, table.columnCount() - 1),
                                )
                                if widget and isinstance(widget, QDoubleSpinBox):
                                    row.append(widget.value())
                                else:
                                    row.append(0.0)
                            values.append(row)

                        inputs[input_name] = np.array(values, dtype=np.float32)

                inputs_list = [inputs]
                frequency_points = np.array([])

            self.infer_button.setEnabled(False)
            self.inference_progress.setRange(0, 0)
            self.inference_progress.setVisible(True)
            self.inference_status.setText("Running inference...")

            self.inference_worker = InferenceWorkerThread(
                self.inference_engine, inputs_list, frequency_points
            )
            self.inference_worker.finished.connect(self._on_inference_finished)
            self.inference_worker.start()

        except Exception as e:
            logger.error(f"Error preparing inference: {e}")
            QMessageBox.critical(self, "Error", f"Error preparing inference:\n{str(e)}")
            self.infer_button.setEnabled(True)
            self.inference_progress.setVisible(False)

    @Slot(bool, str, dict, np.ndarray)
    def _on_inference_finished(
        self, success: bool, message: str, results: dict, frequency_array: np.ndarray
    ):
        """Handle inference completion."""
        self.infer_button.setEnabled(True)
        self.inference_progress.setVisible(False)

        if success:
            # Apply postprocessing if geometry is available
            if self.active_geometry_class:
                try:
                    geometry = self.active_geometry_class()

                    # Pass frequency array if available
                    if frequency_array.size > 0:
                        processed_results = geometry.postprocess_outputs(
                            results, frequency_array
                        )
                    else:
                        processed_results = geometry.postprocess_outputs(results)

                    # Only use processed results if they are not None
                    if processed_results is not None:
                        results = processed_results
                except Exception as e:
                    logger.error(f"Error postprocessing outputs: {e}")
                    # Continue with unprocessed results

            self.results_table.setRowCount(0)

            # Handle case where results might be None
            if results is None:
                self.inference_status.setText(
                    "⚠ Inference completed but no results returned"
                )
                return

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


def create_inference_tab(
    parent: QWidget | None = None, geometry_registry: dict | None = None
) -> InferenceTab:
    """Factory retained for backward compatibility."""
    return InferenceTab(geometry_registry=geometry_registry, parent=parent)
