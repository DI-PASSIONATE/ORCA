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
		self.simulation_tab = SimulationTrainingTab(self)
		self.tabs.addTab(self.simulation_tab, "Simulation + Training")
		
		# Mode 2: Inference & Prediction
		self.inference_tab = InferenceTab(self)
		self.tabs.addTab(self.inference_tab, "Inference + Prediction")
		
		main_layout.addWidget(self.tabs)
		
		self.setLayout(main_layout)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
