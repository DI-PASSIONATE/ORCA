import os
import sys
import importlib.metadata
from PySide6.QtWidgets import (
	QApplication,
	QWidget,
	QLabel,
	QVBoxLayout,
	QHBoxLayout,
	QTabWidget,
	QComboBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from orca.gui.themes import SageTheme, LightTheme
from orca.gui.simulate_train_gui import SimulationTrainingTab
from orca.gui.inference_gui import InferenceTab
from orca.geometry.base_geometry import BaseGeometry
from orca.utils.class_finder import discover_classes


class MainWindow(QWidget):
	def __init__(self):
		super().__init__()
		
		# Set window title to name + version
		self.setWindowTitle(f"ORCA v{importlib.metadata.version('orca')}")
		self.setMinimumSize(800, 600)
		
		self.geometry_registry = self._load_geometries()
		self.current_geometry = next(iter(self.geometry_registry), "")
		
		# Initialize available themes
		self.themes = {
			"Blue": LightTheme(),
			"Sage": SageTheme(),
		}
		
		# Apply theme
		self.set_theme("Blue")
		
		# Setup UI
		self.init_ui()
	
	def set_theme(self, theme_name: str):
		"""Change the current theme"""
		if theme_name in self.themes:
			self.current_theme = self.themes[theme_name]
			self.setPalette(self.current_theme.get_palette())
			self.setStyleSheet(self.current_theme.get_stylesheet())

	def _load_geometries(self):
		"""Load available geometry classes from presets folder for sharing across tabs."""
		presets_dir = os.path.abspath(
			os.path.join(os.path.dirname(__file__), "..", "geometry", "presets")
		)
		return discover_classes(
			base_class=BaseGeometry,
			search_dir=presets_dir,
			module_prefix="orca.geometry.presets",
			extract_default_params=True,
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

		geometry_layout = QHBoxLayout()
		geometry_label = QLabel("Geometry:")
		geometry_label.setStyleSheet("font-weight: bold; padding-right: 5px; color: #EBF4DD;")
		self.geometry_combo = QComboBox()
		self.geometry_combo.addItems(list(self.geometry_registry.keys()))
		self.geometry_combo.currentTextChanged.connect(self.on_geometry_changed)
		self.geometry_combo.setMaximumWidth(200)
		if self.current_geometry:
			index = self.geometry_combo.findText(self.current_geometry)
			if index >= 0:
				self.geometry_combo.setCurrentIndex(index)
		geometry_layout.addWidget(geometry_label)
		geometry_layout.addWidget(self.geometry_combo)
		top_bar.addLayout(geometry_layout)
		
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
		self.simulation_tab = SimulationTrainingTab(self.geometry_registry, self)
		self.tabs.addTab(self.simulation_tab, "Simulation + Training")
		
		# Mode 2: Inference & Prediction
		self.inference_tab = InferenceTab(self)
		self.tabs.addTab(self.inference_tab, "Inference + Prediction")
		
		main_layout.addWidget(self.tabs)
		
		self.setLayout(main_layout)
		self.on_geometry_changed(self.geometry_combo.currentText())

	def on_geometry_changed(self, geometry_name: str):
		"""Propagate geometry selection to all tabs."""
		self.current_geometry = geometry_name
		if hasattr(self, "simulation_tab"):
			self.simulation_tab.set_geometry(geometry_name)
		if hasattr(self, "inference_tab"):
			self.inference_tab.set_active_geometry(geometry_name)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
