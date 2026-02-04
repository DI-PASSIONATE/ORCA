import sys
from PySide6.QtWidgets import QApplication, QDialog
from orca.gui.theme import apply_theme
from orca.gui.welcome_dialog import WelcomeDialog
from orca.gui.pipeline_window import PipelineWindow
from orca.gui.inference_window import InferenceWindow

def run_gui():
    app = QApplication(sys.argv)
    apply_theme(app)
    
    # 1. Show Welcome Dialog
    welcome = WelcomeDialog()
    if welcome.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
        
    # 2. Launch selected window
    if welcome.selected_mode == "pipeline":
        window = PipelineWindow()
    else:
        window = InferenceWindow()
        
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
