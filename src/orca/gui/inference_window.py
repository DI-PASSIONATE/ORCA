from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

class InferenceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORCA Inference")
        self.resize(600, 400)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)
        
        label = QLabel("Inference Module\n\n(Coming Soon)")
        label.setStyleSheet("font-size: 24px; color: gray;")
        layout.addWidget(label)
