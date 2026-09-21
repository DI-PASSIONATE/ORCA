from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from orca.geometry.base_geometry import BaseGeometry
from orca.gui.help_texts import tooltip
from orca.gui.theme import manager as theme_manager
from orca.gui.theme import refresh_style
from orca.gui.utils import get_preset_geometries, load_class_from_file


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
        theme = theme_manager()
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.tokens.space_2)

        # Preset combo box next to the custom-file button
        source_layout = QHBoxLayout()
        source_layout.setSpacing(theme.tokens.space_2)
        self.combo_presets = QComboBox()
        self.combo_presets.setToolTip(tooltip("geometry_combo"))
        self.combo_presets.currentIndexChanged.connect(self.on_preset_changed)

        self.btn_load_custom = QPushButton("Load custom file")
        self.btn_load_custom.setToolTip(tooltip("geometry_file_btn"))
        self.btn_load_custom.clicked.connect(self.load_custom_file)
        theme.bind_icon(self.btn_load_custom, "folder-open-outline")

        source_layout.addWidget(self.combo_presets, 1)
        source_layout.addWidget(self.btn_load_custom)
        layout.addRow("Geometry", source_layout)

        # Name override
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Geometry name")
        self.name_input.setToolTip(tooltip("geometry_name_edit"))
        layout.addRow("Name", self.name_input)

        # Status: colour follows the role property, the text says the same thing.
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        layout.addRow(self.lbl_status)
        self.set_status("No geometry loaded", "muted")

    def set_status(self, text: str, role: str = "muted") -> None:
        """Show *text* under the form; *role* is ``muted``, ``success`` or ``error``."""
        self.lbl_status.setText(text)
        self.lbl_status.setProperty("role", role)
        refresh_style(self.lbl_status)

    def load_presets(self):
        self.combo_presets.clear()
        self.combo_presets.addItem("Select a preset…", None)

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
            self, "Open geometry file", "", "Python files (*.py)"
        )
        if file_path:
            self.combo_presets.setCurrentIndex(0) # Reset preset selection
            cls = load_class_from_file(file_path, BaseGeometry)
            if not cls:
                self.set_status(f"No BaseGeometry subclass found in {file_path}", "error")
                return
            self.load_geometry_from_class(cls)

    def load_geometry_from_class(self, cls):
        try:
            # Instantiate with default arguments
            instance = cls()
            self.current_geometry_class = cls
            self.current_geometry_instance = instance
            self.name_input.setText(instance.name)
            self.set_status(f"Loaded {cls.__name__}", "success")
        except Exception as e:  # noqa: BLE001 - user-supplied class: show the error instead of crashing
            self.set_status(f"Could not instantiate {cls.__name__}: {e}", "error")

    def get_geometry(self):
        if not self.current_geometry_instance:
            return None

        # Update name
        new_name = self.name_input.text()
        if new_name:
            self.current_geometry_instance.name = new_name

        return self.current_geometry_instance
