import sys

from PySide6.QtWidgets import QApplication

from orca.gui import theme
from orca.gui.pipeline_window import PipelineWindow


def run_gui():
    app = QApplication(sys.argv)
    app.setOrganizationName("ORCA")
    app.setApplicationName("ORCA")
    theme.manager().apply()
    window = PipelineWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
