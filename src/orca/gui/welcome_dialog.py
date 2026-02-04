from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)
from PySide6.QtCore import Qt

class WelcomeDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to ORCA")
        self.resize(400, 200)
        self.selected_mode = None
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        title = QLabel("Welcome to ORCA\nOpen AI-Assisted RF Circuit Automation")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        btn_layout = QHBoxLayout()
        
        self.btn_pipeline = QPushButton("Run Pipeline")
        self.btn_pipeline.setMinimumHeight(60)
        self.btn_pipeline.setStyleSheet("font-size: 14px;")
        self.btn_pipeline.clicked.connect(lambda: self.select_mode("pipeline"))
        
        self.btn_inference = QPushButton("Inference")
        self.btn_inference.setMinimumHeight(60)
        self.btn_inference.setStyleSheet("font-size: 14px;")
        self.btn_inference.clicked.connect(lambda: self.select_mode("inference"))
        
        btn_layout.addWidget(self.btn_pipeline)
        btn_layout.addWidget(self.btn_inference)
        
        layout.addLayout(btn_layout)
        
    def select_mode(self, mode):
        self.selected_mode = mode
        self.accept()
