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
from orca.gui.themes import DarkBlueTheme, DarkGreenTheme, LightTheme
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
            
            self.finished.emit(True, "ORCA simulation completed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Error during simulation: {str(e)}")
    
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
            "Dark Blue": DarkBlueTheme(),
            "Dark Green": DarkGreenTheme(),
            "Light": LightTheme(),
        }
        # Apply theme
        self.set_theme("Dark Blue")
        
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
        header.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        header.setFont(font)
        header.setStyleSheet("color: #4CAF50; padding: 10px;")
        top_bar.addWidget(header, 1)
        
        # Theme selector
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("font-weight: bold; padding-right: 5px;")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.themes.keys())
        self.theme_combo.currentTextChanged.connect(self.set_theme)
        self.theme_combo.setMaximumWidth(150)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        top_bar.addLayout(theme_layout)
        
        main_layout.addLayout(top_bar)
        
        # Tab widget for modes
        self.tabs = QTabWidget()
        
        # Mode 1: Simulation & Training
        self.simulation_tab = self.create_simulation_tab()
        self.tabs.addTab(self.simulation_tab, "Simulation + Training")
        
        # Mode 2: Inference & Prediction (placeholder)
        self.inference_tab = self.create_inference_tab()
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
            color = "#9AA0A6"
        elif state == "active":
            icon_label.setText("▶")
            color = "#2A82DA"
        else:  # todo
            icon_label.setText("•")
            color = "#6B7078"

        icon_label.setStyleSheet(f"color: {color}; font-weight: bold; padding-right: 4px;")
        text_label.setText(label)
        text_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        if total and total > 0:
            count_label.setText(f"{current}/{total}")
        else:
            count_label.setText("–/–")
        count_label.setStyleSheet(f"color: {color}; font-family: monospace;")
    
    def create_simulation_tab(self):
        """Create the simulation and training tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Scroll area for the form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Geometry Configuration Group
        geometry_group = QGroupBox("Geometry Configuration")
        self.params_layout = QFormLayout()
        
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(self.geometry_registry.keys())
        self.geometry_combo.currentTextChanged.connect(self.on_geometry_changed)
        self.params_layout.addRow("Geometry Type:", self.geometry_combo)
        
        geometry_group.setLayout(self.params_layout)
        scroll_layout.addWidget(geometry_group)
        
        # Simulation Configuration Group
        sim_config_group = QGroupBox("Simulation Configuration")
        sim_config_layout = QFormLayout()
        
        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setMinimum(1)
        self.num_samples_spin.setMaximum(10000000)
        self.num_samples_spin.setValue(5)
        self.num_samples_spin.setToolTip("Number of geometry samples to generate and simulate")
        sim_config_layout.addRow("Number of Samples:", self.num_samples_spin)
        
        self.cpu_cores_spin = QSpinBox()
        self.cpu_cores_spin.setMinimum(1)
        self.cpu_cores_spin.setMaximum(multiprocessing.cpu_count())
        self.cpu_cores_spin.setValue(min(16, multiprocessing.cpu_count()))
        self.cpu_cores_spin.setToolTip(f"CPU cores to use (max: {multiprocessing.cpu_count()})")
        sim_config_layout.addRow("CPU Cores:", self.cpu_cores_spin)
        
        self.palace_exec_edit = QLineEdit()
        self.palace_exec_edit.setText("apptainer exec ~/Documents/git/palace/palace.sif palace")
        self.palace_exec_edit.setToolTip("Palace executable command")
        sim_config_layout.addRow("Palace Executable:", self.palace_exec_edit)
        
        sim_config_group.setLayout(sim_config_layout)
        scroll_layout.addWidget(sim_config_group)
        
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Model Training group
        model_group = QGroupBox("Model Training")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        # Will be populated when geometry is selected
        
        model_layout.addWidget(QLabel("Select Model Architecture:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        # Status label
        self.step_label = QLabel("Ready to run simulation")
        self.step_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4CAF50;
            padding: 5px;
        """)
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.step_label)

        # Pipeline steps list
        steps_container = QVBoxLayout()
        self.step_items = {}
        for key, label in self.step_order:
            row = QHBoxLayout()
            icon = QLabel("•")
            icon.setFixedWidth(16)
            text = QLabel(label)
            count = QLabel("–/–")
            count.setFixedWidth(70)
            count.setAlignment(Qt.AlignRight)
            row.addWidget(icon)
            row.addWidget(text)
            row.addStretch(1)
            row.addWidget(count)
            steps_container.addLayout(row)
            self.step_items[key] = {"icon": icon, "text": text, "count": count}

        progress_layout.addLayout(steps_container)

        # Progress bar with percentage (current step)
        progress_bar_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m)")
        progress_bar_layout.addWidget(self.progress_bar)
        progress_layout.addLayout(progress_bar_layout)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Run button
        self.run_button = QPushButton("Run ORCA Simulation")
        self.run_button.clicked.connect(self.run_orca_simulation)
        self.run_button.setMinimumHeight(40)
        layout.addWidget(self.run_button)
        
        tab.setLayout(layout)
        
        # Initialize with first geometry
        self.on_geometry_changed(self.geometry_combo.currentText())
        
        # Populate model dropdown with available models
        self.update_model_combo()
        self.reset_step_progress()
        
        return tab
    
    def create_inference_tab(self):
        """Create the inference and prediction tab (placeholder)"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Placeholder content
        placeholder_group = QGroupBox("Inference Mode")
        placeholder_layout = QVBoxLayout()
        
        placeholder_label = QLabel(
            "Inference & Prediction Mode\n\n"
            "This feature is under development and will allow you to:\n\n"
            "• Load trained models\n"
            "• Predict RFIC geometries based on desired specifications\n"
            "• Visualize predicted vs. actual performance\n"
            "• Export optimized geometries\n\n"
            "Stay tuned for updates!"
        )
        placeholder_label.setWordWrap(True)
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("font-size: 14px; padding: 40px;")
        
        placeholder_layout.addWidget(placeholder_label)
        placeholder_group.setLayout(placeholder_layout)
        layout.addWidget(placeholder_group)
        
        tab.setLayout(layout)
        return tab
    
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
            color: #2A82DA;
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
        color = "#2A82DA"
        if self.step_state.get(step, {}).get("state") == "done":
            color = "#4CAF50"
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
                color: #4CAF50;
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
