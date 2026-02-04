from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QScrollArea, QLabel, QProgressBar, QMessageBox, QTextEdit
)
from PySide6.QtCore import QThread, Signal, Qt

from orca import ORCA
from orca.gui.utils import get_available_stages
from orca.gui.widgets.geometry_selector import GeometrySelector
from orca.gui.widgets.stage_widget import StageConfigWidget
from orca.logger import logger
import logging

class LogSignalHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class PipelineWorker(QThread):
    progress = Signal(str, int, int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, orca_instance, geometry):
        super().__init__()
        self.orca = orca_instance
        self.geometry = geometry

    def run(self):
        try:
            self.orca.run(
                geometry=self.geometry, 
                progress_callback=self.progress_callback
            )
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())

    def progress_callback(self, stage_name, current, total, message):
        self.progress.emit(stage_name, current, total, message)

class PipelineWindow(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORCA Pipeline")
        self.resize(1000, 800)
        
        self.stages_widgets = []
        self.init_ui()
        self.setup_logging()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left Panel: Configuration
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        config_scroll.setWidget(config_widget)
        
        # Geometry Section
        config_layout.addWidget(QLabel("<h2>1. Geometry Selection</h2>"))
        self.geometry_selector = GeometrySelector()
        config_layout.addWidget(self.geometry_selector)
        
        # Stages Section
        config_layout.addWidget(QLabel("<h2>2. Pipeline Stages</h2>"))
        config_layout.addWidget(QLabel("Select active stages and configure parameters:"))
        
        available_stages = get_available_stages()
        for stage_cls in available_stages:
            sw = StageConfigWidget(stage_cls)
            self.stages_widgets.append(sw)
            config_layout.addWidget(sw)
            
        config_layout.addStretch()
        
        # Right Panel: Logs and button
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Logs will appear here...")
        
        # Progress bar that uses the callbacks
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        
        # Run Button
        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_pipeline)
        
        # Right side: "Console"-style log output and progress
        right_layout.addWidget(QLabel("<h2>Logs & Status</h2>"))
        right_layout.addWidget(self.log_output)
        right_layout.addWidget(self.progress_label)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.btn_run)
        
        # Add sub-widgets to main layout
        main_layout.addWidget(config_scroll, 1)
        main_layout.addWidget(right_panel, 1)

    def setup_logging(self):
        handler = LogSignalHandler(self.log_signal)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s >> %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler) # Add a custom handler to the ORCA logger to avoid duplicate logs
        self.log_signal.connect(self.log_output.append)

    def run_pipeline(self):
        geometry = self.geometry_selector.get_geometry()
        if not geometry:
            QMessageBox.warning(self, "Invalid Geometry", "Please select a valid geometry first.")
            return
            
        stages = []
        for sw in self.stages_widgets:
            stage_instance = sw.get_instance()
            if stage_instance:
                stages.append(stage_instance)
                
        if not stages:
            QMessageBox.warning(self, "No Stages", "Please select at least one pipeline stage.")
            return
            
        # Instantiate ORCA
        orca_instance = ORCA(stages)
        
        # Disable button
        self.btn_run.setEnabled(False)
        self.log_output.clear()
        self.log_output.append("Starting pipeline...")
        
        self.worker = PipelineWorker(orca_instance, geometry)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.pipeline_finished)
        self.worker.error.connect(self.pipeline_error)
        self.worker.start()
        
    def update_progress(self, stage_name, current, total, message):
        self.progress_label.setText(f"{stage_name}: {message}")
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        else:
            self.progress_bar.setValue(0)
            
    def pipeline_finished(self):
        self.btn_run.setEnabled(True)
        self.progress_label.setText("Pipeline Completed Successfully")
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Success", "ORCA Pipeline finished successfully!")
        
    def pipeline_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.progress_label.setText("Error Occurred")
        self.log_output.append(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred:\n{error_msg}")
        
    def closeEvent(self, event):
        # Clean up logger handler
        logging.getLogger().handlers = [h for h in logging.getLogger().handlers if not isinstance(h, LogSignalHandler)]
        event.accept()
