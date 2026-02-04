from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

def apply_theme(app: QApplication):
    app.setStyle("Basic") # Fusion style automatically switched dark/light themes
    
    palette = QPalette()
    
    # Blue color palette for "ORCA"
    dark_blue = QColor(10, 25, 47)
    medium_blue = QColor(23, 42, 69)
    accent_color = QColor(250, 170, 80) # Orange-ish accent
    text_color = QColor(204, 214, 246)
    secondary_text = QColor(136, 146, 176)
    
    # Window background
    palette.setColor(QPalette.ColorRole.Window, dark_blue)
    palette.setColor(QPalette.ColorRole.WindowText, text_color)
    
    # Base (text info backgrounds)
    palette.setColor(QPalette.ColorRole.Base, medium_blue)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark_blue)
    palette.setColor(QPalette.ColorRole.ToolTipBase, medium_blue)
    palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
    palette.setColor(QPalette.ColorRole.Text, text_color)
    
    # Buttons
    palette.setColor(QPalette.ColorRole.Button, medium_blue)
    palette.setColor(QPalette.ColorRole.ButtonText, text_color)
    palette.setColor(QPalette.ColorRole.BrightText, accent_color)
    
    # Highlight
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    # Links
    palette.setColor(QPalette.ColorRole.Link, accent_color)
    palette.setColor(QPalette.ColorRole.LinkVisited, accent_color)

    app.setPalette(palette)
    
    # Additional stylesheets for specific widgets
    app.setStyleSheet(f"""
        QMainWindow {{
            background-color: {dark_blue.name()};
        }}
        QGroupBox {{
            border: 1px solid {secondary_text.name()};
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
            color: {accent_color.name()};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
        }}
        QPushButton {{
            background-color: {medium_blue.name()};
            border: 1px solid {secondary_text.name()};
            border-radius: 4px;
            padding: 5px 15px;
            color: {text_color.name()};
        }}
        QPushButton:hover {{
            background-color: {QColor(35, 55, 85).name()};
            border-color: {accent_color.name()};
        }}
        QPushButton:pressed {{
            background-color: {dark_blue.name()};
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {medium_blue.name()};
            border: 1px solid {secondary_text.name()};
            border-radius: 3px;
            padding: 3px;
            color: {text_color.name()};
        }}
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border: 1px solid {accent_color.name()};
        }}
        QCheckBox {{
            color: {text_color.name()};
        }}
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border: 1px solid {secondary_text.name()};
            background: {medium_blue.name()};
        }}
        QCheckBox::indicator:checked {{
            background: {accent_color.name()};
            border: 1px solid {accent_color.name()};
        }}
        QLabel {{
            color: {text_color.name()};
        }}
    """)
