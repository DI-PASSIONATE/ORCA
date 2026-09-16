import sys

from PySide6.QtWidgets import QApplication

from orca.gui.pipeline_window import PipelineWindow
from orca.gui.theme import apply_theme


def run_gui():
    app = QApplication(sys.argv)

    window = PipelineWindow()
    apply_theme(window)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
