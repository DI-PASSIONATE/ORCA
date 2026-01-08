import sys
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
import torch.nn as nn
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSpinBox,
    QTabWidget,
    QComboBox,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QMessageBox,
    QFileDialog,
    QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

from orca.orca import ORCA
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.gui.themes import SageTheme, LightTheme
from orca.gui.simulate_train_gui import create_simulation_tab
from orca.gui.inference_gui import create_inference_tab
from orca.utils.class_finder import discover_classes


class ORCAWorkerThread(QThread):
    """Background thread for running ORCA simulations"""
    progress = Signal(str, int, int, str)  # step, current, total, message
    finished = Signal(bool, str)
    
    def __init__(self, geometry_class, geometry_params, num_samples, cpu_cores, palace_executable):
        super().__init__()
        self.geometry_class = geometry_class
        self.geometry_params = geometry_params
        self.num_samples = num_samples
        self.cpu_cores = cpu_cores
        self.palace_executable = palace_executable
    
    def run(self):
        try:
            self.progress.emit("Initializing", 0, 0, "Initializing geometry...")
            
            # Create geometry instance
            geometry = self.geometry_class(**self.geometry_params)
            
            orca_instance = ORCA(geometry)
            
            # Run with progress callback
            orca_instance.run(
                num_samples=self.num_samples,
                cpu_cores=self.cpu_cores,
                palace_executable=self.palace_executable,
                progress_callback=self.on_progress_update
            )
            
            self.finished.emit(True, "ORCA training completed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Error during simulation/training: {str(e)}")
    
    def on_progress_update(self, step, current, total, message):
        """Callback for ORCA progress updates"""
        self.progress.emit(step, current, total, message)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set window title to name + version
        self.setWindowTitle(f"ORCA v{importlib.metadata.version('orca')}")
        self.setMinimumSize(800, 600)
        
        # Initialize available themes
        self.themes = {
            "Blue": LightTheme(),
            "Sage": SageTheme(),
        }
        # Predeclare UI elements for type checkers
        self.geometry_combo = None
        self.params_layout = None
        self.model_combo = None
        self.num_samples_spin = None
        self.cpu_cores_spin = None
        self.palace_exec_edit = None
        self.stackup_xml_edit = None
        self.simconfig_edit = None
        self.geometry_name_edit = None
        self.run_button = None
        self.progress_bar = None
        self.step_label = None
        # Apply theme
        self.set_theme("Blue")
        
        # Initialize geometry registry
        self.geometry_registry = self.load_geometries()
        
        # Initialize model registry
        self.model_registry = self.load_models()

        # Define pipeline steps for progress display
        self.step_order = [
            ("GDS Generation", "Generate GDS"),
            ("Palace Conversion", "Convert to Palace"),
            ("Palace Simulation", "Run Palace simulations"),
            ("Model Training", "Train model"),
            ("Model Evaluation", "Evaluate model"),
        ]
        self.step_items = {}
        self.step_state = {}
        
        # Setup UI
        self.init_ui()
    
    def set_theme(self, theme_name: str):
        """Change the current theme"""
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
            self.setPalette(self.current_theme.get_palette())
            self.setStyleSheet(self.current_theme.get_stylesheet())
    
    def load_geometries(self):
        """Load available geometry classes from presets folder using class_finder"""
        presets_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), 
                "..", "geometry", "presets"
            )
        )
        
        return discover_classes(
            base_class=BaseGeometry,
            search_dir=presets_dir,
            module_prefix="orca.geometry.presets",
            extract_default_params=True
        )
    
    def load_models(self):
        """Load available model classes from training/models folder using class_finder"""
        models_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "training", "models"
            )
        )
        
        return discover_classes(
            base_class=nn.Module,
            search_dir=models_dir,
            module_prefix="orca.training.models",
            extract_default_params=False
        )
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # Top bar with header and theme selector
        top_bar = QHBoxLayout()
        
        # Header with logo text
        header = QLabel("ORCA - Open RF Circuit Automation")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        header.setFont(font)
        header.setStyleSheet("color: #EBF4DD; padding: 10px;")
        top_bar.addWidget(header, 1)
        
        # Theme selector
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("font-weight: bold; padding-right: 5px; color: #EBF4DD;")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(self.themes.keys()))
        self.theme_combo.currentTextChanged.connect(self.set_theme)
        self.theme_combo.setMaximumWidth(150)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        top_bar.addLayout(theme_layout)
        
        main_layout.addLayout(top_bar)
        
        # Tab widget for modes
        self.tabs = QTabWidget()
        
        # Mode 1: Simulation & Training
        self.simulation_tab = create_simulation_tab(self)
        self.tabs.addTab(self.simulation_tab, "Simulation + Training")
        
        # Mode 2: Inference & Prediction (placeholder)
        self.inference_tab = create_inference_tab(self)
        self.tabs.addTab(self.inference_tab, "Inference + Prediction")
        
        main_layout.addWidget(self.tabs)
        
        self.setLayout(main_layout)

    def reset_step_progress(self):
        """Reset tracking state for all pipeline steps"""
        self.step_state = {
            key: {
                "state": "todo",  # todo | active | done
                "current": 0,
                "total": 0
            }
            for key, _ in self.step_order
        }
        self.refresh_step_widgets()

    def refresh_step_widgets(self, active_step: str | None = None):
        """Refresh the visual status for all steps."""
        step_keys = [k for k, _ in self.step_order]
        idx_active = step_keys.index(active_step) if active_step in step_keys else None

        for key, label in self.step_order:
            data = self.step_state.get(key, {})
            state = data.get("state", "todo")
            current = data.get("current", 0)
            total = data.get("total", 0)

            # If a new active step is specified, mark previous active steps as done
            if idx_active is not None:
                idx_this = step_keys.index(key)
                if idx_this < idx_active:
                    state = "done"
                elif idx_this > idx_active and state == "active":
                    state = "todo"

            self.update_step_widget(key, label, state, current, total)

    def update_step_widget(self, key: str, label: str, state: str, current: int, total: int):
        """Update a single step widget based on state and counts."""
        widget = self.step_items.get(key)
        if not widget:
            return

        icon_label = widget["icon"]
        text_label = widget["text"]
        count_label = widget["count"]

        if state == "done":
            icon_label.setText("✓")
            color = "#8FDBFF"  # sage
        elif state == "active":
            icon_label.setText("▶")
            color = "#FF9532"  # deep sage
        else:  # todo
            icon_label.setText("•")
            color = "#7D8087"

        icon_label.setStyleSheet(f"color: {color}; font-weight: bold; padding-right: 4px;")
        text_label.setText(label)
        text_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        if total and total > 0:
            count_label.setText(f"{current}/{total}")
        else:
            count_label.setText("–/–")
        count_label.setStyleSheet(f"color: {color}; font-family: monospace;")
    
    def on_geometry_changed(self, geometry_name):
        """Handle geometry selection change"""
        # Clear existing parameter inputs (but keep the geometry type row)
        while self.params_layout.rowCount() > 1:
            self.params_layout.removeRow(1)
        
        # Get selected geometry info
        geometry_info = self.geometry_registry.get(geometry_name)
        if not geometry_info:
            return
        
        # Update model combo to show default model for this geometry
        self.update_model_combo()
        
        # Add parameter inputs based on geometry class
        # For now, we'll add common parameters like stackup_xml and simconfig_filename
        
        self.stackup_xml_edit = QLineEdit()
        default_stackup = geometry_info.get("default_params", {}).get("stackup_xml")
        if default_stackup:
            self.stackup_xml_edit.setText(os.path.expanduser(str(default_stackup)))
        self.stackup_xml_edit.setToolTip("Path to stackup XML file")
        
        browse_stackup_btn = QPushButton("Browse...")
        browse_stackup_btn.clicked.connect(lambda: self.browse_file(self.stackup_xml_edit, "XML Files (*.xml)"))
        
        stackup_layout = QHBoxLayout()
        stackup_layout.addWidget(self.stackup_xml_edit)
        stackup_layout.addWidget(browse_stackup_btn)
        self.params_layout.addRow("Stackup XML:", stackup_layout)
        
        self.simconfig_edit = QLineEdit()
        default_simconfig = geometry_info.get("default_params", {}).get("simconfig_filename")
        if default_simconfig:
            self.simconfig_edit.setText(os.path.expanduser(str(default_simconfig)))

        self.simconfig_edit.setToolTip("Path to simulation config file")
        
        browse_simconfig_btn = QPushButton("Browse...")
        browse_simconfig_btn.clicked.connect(lambda: self.browse_file(self.simconfig_edit, "SimConfig Files (*.simcfg)"))
        
        simconfig_layout = QHBoxLayout()
        simconfig_layout.addWidget(self.simconfig_edit)
        simconfig_layout.addWidget(browse_simconfig_btn)
        self.params_layout.addRow("SimConfig File:", simconfig_layout)
        
        # Add geometry name field
        self.geometry_name_edit = QLineEdit()
        default_name = geometry_info.get("default_params", {}).get("name")
        if default_name:
            self.geometry_name_edit.setText(str(default_name))
        self.geometry_name_edit.setToolTip("Name for the geometry")
        self.params_layout.addRow("Geometry Name:", self.geometry_name_edit)
    
    def update_model_combo(self):
        """Update model combo box with available models and highlight default"""
        self.model_combo.clear()
        
        # Get the selected geometry
        geometry_name = self.geometry_combo.currentText()
        geometry_info = self.geometry_registry.get(geometry_name)
        
        # Try to get the default model from the geometry
        default_model_class = None
        if geometry_info:
            try:
                # Instantiate geometry to call create_model()
                geometry_class = geometry_info["class"]
                temp_geometry = geometry_class()
                default_model = temp_geometry.create_model()
                default_model_class = type(default_model).__name__
            except Exception as e:
                logger.warning(f"Could not get default model for {geometry_name}: {e}")
        
        # Add all available models
        model_names = []
        default_index = 0
        for i, (display_name, model_info) in enumerate(self.model_registry.items()):
            class_name = model_info["class_name"]
            model_names.append(display_name)
            
            # Check if this is the default model
            if default_model_class and class_name == default_model_class:
                default_index = i
        
        if model_names:
            self.model_combo.addItems(model_names)
            self.model_combo.setCurrentIndex(default_index)
        else:
            self.model_combo.addItem("No models found")
            logger.warning("No models discovered in training/models folder")
    
    def browse_file(self, line_edit, file_filter):
        """Open file browser dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File", 
            os.path.expanduser("~"),
            file_filter
        )
        if filename:
            line_edit.setText(filename)
    
    def run_orca_simulation(self):
        """Run ORCA simulation in background thread"""
        # Validate inputs
        geometry_name = self.geometry_combo.currentText()
        geometry_info = self.geometry_registry.get(geometry_name)
        
        if not geometry_info:
            QMessageBox.warning(self, "Error", "Please select a valid geometry")
            return
        
        # Gather parameters
        geometry_params = {
            "name": self.geometry_name_edit.text(),
            "stackup_xml": os.path.expanduser(self.stackup_xml_edit.text()),
            "simconfig_filename": os.path.expanduser(self.simconfig_edit.text())
        }
        
        # Validate files exist
        if not os.path.exists(geometry_params["stackup_xml"]):
            QMessageBox.warning(self, "Error", f"Stackup XML file not found:\n{geometry_params['stackup_xml']}")
            return
        
        if not os.path.exists(geometry_params["simconfig_filename"]):
            QMessageBox.warning(self, "Error", f"SimConfig file not found:\n{geometry_params['simconfig_filename']}")
            return
        
        num_samples = self.num_samples_spin.value()
        cpu_cores = self.cpu_cores_spin.value()
        palace_executable = self.palace_exec_edit.text()
        
        # Disable run button during execution
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, num_samples)
        self.progress_bar.setValue(0)
        self.step_label.setText("Initializing...")
        self.step_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #5A7863;
            padding: 5px;
        """)
        self.reset_step_progress()
        if self.step_order:
            first_key, _ = self.step_order[0]
            self.step_state[first_key]["state"] = "active"
            self.step_state[first_key]["total"] = num_samples
            self.refresh_step_widgets(active_step=first_key)
        
        # Create and start worker thread
        self.worker = ORCAWorkerThread(
            geometry_info["class"],
            geometry_params,
            num_samples,
            cpu_cores,
            palace_executable
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    @Slot(str, int, int, str)
    def on_progress(self, step, current, total, message):
        """Update progress display with detailed step information"""
        # Update headline text
        self.step_label.setText(message)

        # Handle overall completion
        if step == "Complete":
            for key in self.step_state:
                self.step_state[key]["state"] = "done"
                if total:
                    self.step_state[key]["total"] = total
                    self.step_state[key]["current"] = total
            self.refresh_step_widgets()
            self.progress_bar.setVisible(False)
            return

        # Handle unknown steps (e.g., Initializing)
        step_keys = [k for k, _ in self.step_order]
        if step not in step_keys:
            self.progress_bar.setRange(0, 0)  # Indeterminate for setup
            self.progress_bar.setVisible(True)
            return

        # Update internal state
        idx_active = step_keys.index(step)
        for i, key in enumerate(step_keys):
            if i < idx_active:
                self.step_state[key]["state"] = "done"
            elif i == idx_active:
                self.step_state[key]["state"] = "active"
                self.step_state[key]["current"] = current
                if total:
                    self.step_state[key]["total"] = total
                # If finished, mark done
                if total and current >= total:
                    self.step_state[key]["state"] = "done"
            else:
                # Keep done if already done, otherwise todo
                if self.step_state[key].get("state") != "done":
                    self.step_state[key]["state"] = "todo"

        # Refresh visuals
        self.refresh_step_widgets(active_step=step)

        # Update progress bar for current step
        if total and total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(current, total))
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(True)

        # Color headline based on state
        color = "#5A7863"  # deep sage for active/default
        if self.step_state.get(step, {}).get("state") == "done":
            color = "#90AB8B"  # sage for success
        if "failed" in message.lower() or "error" in message.lower():
            color = "#FF5252"
        self.step_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {color};
            padding: 5px;
        """)
    
    @Slot(bool, str)
    def on_finished(self, success, message):
        """Handle simulation completion"""
        self.run_button.setEnabled(True)
        
        if success:
            # Mark all steps as done
            for key in self.step_state:
                self.step_state[key]["state"] = "done"
            self.refresh_step_widgets()
            self.progress_bar.setVisible(False)
            self.step_label.setText("✅ Simulation Completed Successfully")
            self.step_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: #90AB8B;
                padding: 5px;
            """)
            QMessageBox.information(self, "Success", message)
        else:
            self.progress_bar.setVisible(False)
            self.step_label.setText("❌ Simulation Failed")
            self.step_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: #FF5252;
                padding: 5px;
            """)
            QMessageBox.critical(self, "Error", message)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
