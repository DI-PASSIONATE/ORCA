from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox
)
from orca.gui.utils import get_preset_geometries, load_class_from_file
from orca.geometry.base_geometry import BaseGeometry

class GeometrySelector(QWidget):
    """
    This widget allows users to select a geometry either from preset options
    or by loading a custom Python file defining a BaseGeometry subclass.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_geometry_class = None
        self.current_geometry_instance = None
        
        self.init_ui()
        self.load_presets()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Selection Mode
        mode_layout = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.currentIndexChanged.connect(self.on_preset_changed)
        
        self.btn_load_custom = QPushButton("Load Custom .py")
        self.btn_load_custom.clicked.connect(self.load_custom_file)
        
        layout.addWidget(QLabel("Select Geometry:"))
        layout.addLayout(mode_layout)
        mode_layout.addWidget(self.combo_presets)
        mode_layout.addWidget(self.btn_load_custom)
        
        # Name Override
        name_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Geometry Name")
        
        name_layout.addWidget(QLabel("Geometry Name:"))
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        self.lbl_status = QLabel("No geometry loaded")
        layout.addWidget(self.lbl_status)
        
    def load_presets(self):
        self.combo_presets.clear()
        self.combo_presets.addItem("Select a preset...", None)
        
        presets = get_preset_geometries()
        for cls in presets:
            self.combo_presets.addItem(cls.__name__, cls)
            
    def on_preset_changed(self, index):
        if index == 0:
            return
            
        cls = self.combo_presets.currentData()
        if cls and issubclass(cls, BaseGeometry):
            self.load_geometry_from_class(cls)
            
    def load_custom_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry File", "", "Python Files (*.py)"
        )
        if file_path:
            self.combo_presets.setCurrentIndex(0) # Reset preset selection
            cls = load_class_from_file(file_path, BaseGeometry)
            if not cls:
                QMessageBox.warning(self, "Error", f"Could not find valid BaseGeometry subclass in {file_path}")
                self.lbl_status.setText("Error loading geometry")
                return
            self.load_geometry_from_class(cls)
            
    def load_geometry_from_class(self, cls):
        try:
            # Instantiate with default arguments
            instance = cls()
            self.current_geometry_class = cls
            self.current_geometry_instance = instance
            self.name_input.setText(instance.name)
            self.lbl_status.setText(f"Loaded: {cls.__name__}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to instantiate geometry class: {e}")
            self.lbl_status.setText("Error instantiating class")

    def get_geometry(self):
        if not self.current_geometry_instance:
            return None
            
        # Update name
        new_name = self.name_input.text()
        if new_name:
            self.current_geometry_instance.name = new_name
            
        return self.current_geometry_instance
