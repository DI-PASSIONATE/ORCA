import multiprocessing

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
	QWidget,
	QVBoxLayout,
	QScrollArea,
	QGroupBox,
	QFormLayout,
	QComboBox,
	QSpinBox,
	QLineEdit,
	QPushButton,
	QLabel,
	QHBoxLayout,
	QProgressBar,
)


def create_simulation_tab(self):
	"""Create the simulation and training tab."""
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
	self.geometry_combo.addItems(list(self.geometry_registry.keys()))
	self.geometry_combo.currentTextChanged.connect(self.on_geometry_changed)
	self.params_layout.addRow("Geometry Type:", self.geometry_combo)

	geometry_group.setLayout(self.params_layout)
	scroll_layout.addWidget(geometry_group)

	# Simulation Configuration Group
	sim_config_group = QGroupBox("Simulation Configuration")
	sim_config_layout = QFormLayout()

	self.num_samples_spin = QSpinBox()
	self.num_samples_spin.setMinimum(0)
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
	model_layout = QFormLayout()

	self.model_combo = QComboBox()
	# Will be populated when geometry is selected

	model_layout.addRow(QLabel("Select Model Architecture:"), self.model_combo)
	model_group.setLayout(model_layout)
	scroll_layout.addWidget(model_group)

	# Progress section
	progress_group = QGroupBox("Progress")
	progress_layout = QVBoxLayout()

	# Status label
	self.step_label = QLabel("Ready to run simulation")
	self.step_label.setStyleSheet(
		"""
			font-size: 14px;
			font-weight: bold;
			color: #EBF4DD;
			padding: 5px;
		"""
	)
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
		count.setAlignment(Qt.AlignmentFlag.AlignRight)
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
