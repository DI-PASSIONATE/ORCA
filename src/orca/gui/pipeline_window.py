import logging

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from orca import ORCA
from orca.gui.help_texts import tooltip
from orca.gui.theme import ThemeTokens, refresh_style
from orca.gui.theme import manager as theme_manager
from orca.gui.utils import get_available_stages
from orca.gui.widgets.geometry_selector import GeometrySelector
from orca.gui.widgets.stage_widget import StageConfigWidget
from orca.logger import logger


class LogSignalHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class PipelineWorker(QThread):
    progress = Signal(str, int, int, str)
    confirm_overwrite = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, orca_instance, geometry):
        super().__init__()
        self.orca = orca_instance
        self.geometry = geometry
        self._overwrite_result = None
        self._waiting_for_overwrite = False

    def run(self):
        try:
            self.orca.run(
                geometry=self.geometry,
                progress_callback=self.progress_callback,
                overwrite_callback=self.overwrite_callback
            )
            self.finished.emit()
        except Exception as e:  # noqa: BLE001 - worker thread: report every failure to the GUI
            import traceback
            self.error.emit(str(e) + "\n" + traceback.format_exc())

    def progress_callback(self, stage_name, current, total, message):
        self.progress.emit(stage_name, current, total, message)

    def overwrite_callback(self, base_dir):
        self._overwrite_result = None
        self._waiting_for_overwrite = True
        self.confirm_overwrite.emit(base_dir)

        while self._waiting_for_overwrite:
            self.msleep(100)

        return self._overwrite_result

    def set_overwrite_result(self, result):
        self._overwrite_result = result
        self._waiting_for_overwrite = False

class PipelineWindow(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._theme = theme_manager()
        self._theme.theme_changed.connect(self._on_theme_changed)

        self.setWindowTitle("ORCA Pipeline")
        self.resize(1000, 800)

        self.stages_widgets = []
        self._log_handler = None
        self.init_ui()
        self.setup_logging()
        self._on_theme_changed(self._theme.tokens)

    def init_ui(self):
        tokens = self._theme.tokens
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(
            tokens.space_4, tokens.space_4, tokens.space_4, tokens.space_4
        )
        root_layout.setSpacing(tokens.space_3)

        # Global controls: run and appearance, shown regardless of the panels below
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(tokens.space_2)

        self.btn_run = QPushButton("Run pipeline")
        self.btn_run.setProperty("primaryAction", True)
        self.btn_run.setProperty("actionState", "start")
        self.btn_run.setFixedSize(220, tokens.control_h_lg)
        self.btn_run.setToolTip(tooltip("run_btn"))
        self.btn_run.clicked.connect(self.run_pipeline)
        self._theme.bind_icon(self.btn_run, "play", on_fill=True, size=24)

        self.theme_btn = QPushButton()
        self.theme_btn.setProperty("flat", True)
        self.theme_btn.setFixedSize(tokens.control_h_lg, tokens.control_h_lg)
        self.theme_btn.setAccessibleName("Appearance")
        self.theme_btn.clicked.connect(self._theme.cycle_mode)
        self._theme.bind_icon(self.theme_btn, "theme-light-dark", tertiary=True, size=24)

        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_run)
        controls_layout.addWidget(self.theme_btn)
        root_layout.addLayout(controls_layout)

        # Progress: the label names the stage, the bar colour names the state
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(tokens.space_2)
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar, stretch=1)
        root_layout.addLayout(progress_layout)
        self.set_status("Ready")

        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(tokens.space_3)
        root_layout.addLayout(panels_layout, stretch=1)

        # Left panel: configuration
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, tokens.space_2, 0)
        config_layout.setSpacing(tokens.space_3)
        config_scroll.setWidget(config_widget)

        # Geometry section
        config_layout.addWidget(self._heading("1. Geometry"))
        geometry_group = QGroupBox()
        geometry_layout = QVBoxLayout(geometry_group)
        geometry_layout.setContentsMargins(
            tokens.space_3, tokens.space_2, tokens.space_3, tokens.space_3
        )
        self.geometry_selector = GeometrySelector()
        geometry_layout.addWidget(self.geometry_selector)
        config_layout.addWidget(geometry_group)

        # Stages section
        config_layout.addWidget(self._heading("2. Pipeline stages"))
        stages_caption = QLabel("Enable the stages to run and set their parameters")
        stages_caption.setProperty("role", "muted")
        config_layout.addWidget(stages_caption)

        available_stages = get_available_stages()
        for stage_cls in available_stages:
            sw = StageConfigWidget(stage_cls)
            self.stages_widgets.append(sw)
            config_layout.addWidget(sw)

        config_layout.addStretch()

        # Right panel: log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(
            tokens.space_3, tokens.space_2, tokens.space_3, tokens.space_3
        )
        self.log_output = QPlainTextEdit()
        self.log_output.setProperty("role", "log")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Pipeline messages appear here")
        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_output.setToolTip(tooltip("log_output"))
        log_layout.addWidget(self.log_output)

        panels_layout.addWidget(config_scroll, 1)
        panels_layout.addWidget(log_group, 1)

    # ------------------------------------------------------------------
    # Appearance and status
    # ------------------------------------------------------------------

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "heading")
        return label

    def _on_theme_changed(self, tokens: ThemeTokens) -> None:
        self.theme_btn.setToolTip(
            tooltip("theme_btn").format(mode=self._theme.mode, theme=tokens.name)
        )

    def set_status(self, text: str, role: str = "muted") -> None:
        """Show *text* next to the progress bar; *role* is ``muted``, ``success`` or ``error``."""
        self.progress_label.setText(text)
        self.progress_label.setProperty("role", role)
        refresh_style(self.progress_label)

    def _set_progress_state(self, state: str) -> None:
        """Colour the progress bar: ``running`` (tide) or ``finished`` (moss)."""
        self.progress_bar.setProperty("progressState", state)
        refresh_style(self.progress_bar)

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_run.setText("Running…" if running else "Run pipeline")
        refresh_style(self.btn_run)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def setup_logging(self):
        handler = LogSignalHandler(self.log_signal)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s >> %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        self._log_handler = handler
        # Fixes ORCA ASCII art being printed in a messed up way in the GUI
        self.log_signal.connect(self.append_log_line)

    def append_log_line(self, msg):
        self.log_output.appendPlainText(msg)

    def run_pipeline(self):
        geometry = self.geometry_selector.get_geometry()
        if not geometry:
            self.set_status("Select a geometry before running", "error")
            self.geometry_selector.set_status("Select a preset or load a custom file", "error")
            return

        stages = []
        for sw in self.stages_widgets:
            stage_instance = sw.get_instance()
            if stage_instance:
                stages.append(stage_instance)

        if not stages:
            self.set_status("Enable at least one pipeline stage", "error")
            return

        # Instantiate ORCA
        orca_instance = ORCA(stages)

        self._set_running(True)
        self._set_progress_state("running")
        self.progress_bar.setValue(0)
        self.set_status("Starting pipeline")
        self.log_output.clear()
        self.log_output.appendPlainText("Starting pipeline...")

        self.worker = PipelineWorker(orca_instance, geometry)
        self.worker.progress.connect(self.update_progress)
        self.worker.confirm_overwrite.connect(self.handle_overwrite_confirmation)
        self.worker.finished.connect(self.pipeline_finished)
        self.worker.error.connect(self.pipeline_error)
        self.worker.start()

    def handle_overwrite_confirmation(self, base_dir):
        # Existing results would be lost, so this is the one place a confirmation is warranted.
        reply = QMessageBox.question(
            self,
            "Overwrite results",
            f"The output directory {base_dir} already exists. Stages may overwrite its files. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        self.worker.set_overwrite_result(reply == QMessageBox.StandardButton.Yes)

    def update_progress(self, stage_name, current, total, message):
        self.set_status(f"{stage_name}: {message}")
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        else:
            self.progress_bar.setValue(0)

    def pipeline_finished(self):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self._set_progress_state("finished")
        self.set_status("Pipeline finished", "success")

    def pipeline_error(self, error_msg):
        self._set_running(False)
        self.set_status("Pipeline failed, see the log", "error")
        self.log_output.appendPlainText(f"ERROR: {error_msg}")

    def closeEvent(self, event):
        if self._log_handler is not None:
            logger.removeHandler(self._log_handler)
            self._log_handler = None
        event.accept()
