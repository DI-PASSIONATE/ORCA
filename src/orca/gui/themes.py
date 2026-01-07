from abc import ABC, abstractmethod
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt


class OrcaTheme(ABC):
    """Abstract base class for ORCA GUI themes."""

    @abstractmethod
    def get_palette(self) -> QPalette:
        """Return the QPalette object for the theme."""

    @abstractmethod
    def get_stylesheet(self) -> str:
        """Return the stylesheet string for the theme."""

    def get_name(self) -> str:
        """Return the display name of the theme."""
        return self.__class__.__name__


class DarkBlueTheme(OrcaTheme):
    """Dark Blue theme - Modern dark blue-green color scheme."""

    def get_palette(self) -> QPalette:
        palette = QPalette()
        
        palette.setColor(QPalette.Window, QColor(30, 40, 50))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(40, 50, 60))
        palette.setColor(QPalette.AlternateBase, QColor(45, 55, 65))
        palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(50, 70, 90))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        return palette

    def get_stylesheet(self) -> str:
        return """QGroupBox {
                border: 2px solid #2A82DA;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #4CAF50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2A82DA;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #1E5FA0;
            }
            QPushButton:pressed {
                background-color: #144270;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #888888;
            }
            QTabWidget::pane {
                border: 2px solid #2A82DA;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #3C4C5C;
                color: #DCDCDC;
                padding: 8px 20px;
                margin: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2A82DA;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #4A5A6A;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3C4C5C;
                border: 1px solid #2A82DA;
                border-radius: 3px;
                padding: 4px;
                color: #DCDCDC;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #4CAF50;
            }
            QTextEdit {
                background-color: #1E2830;
                border: 1px solid #2A82DA;
                border-radius: 3px;
                color: #DCDCDC;
            }
            QProgressBar {
                border: 2px solid #2A82DA;
                border-radius: 5px;
                text-align: center;
                background-color: #3C4C5C;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """


class DarkGreenTheme(OrcaTheme):
    """Dark Green theme - Deep forest green color scheme."""

    def get_palette(self) -> QPalette:
        palette = QPalette()
        
        palette.setColor(QPalette.Window, QColor(20, 40, 30))
        palette.setColor(QPalette.WindowText, QColor(230, 240, 235))
        palette.setColor(QPalette.Base, QColor(30, 50, 40))
        palette.setColor(QPalette.AlternateBase, QColor(35, 55, 45))
        palette.setColor(QPalette.ToolTipBase, QColor(230, 240, 235))
        palette.setColor(QPalette.ToolTipText, QColor(230, 240, 235))
        palette.setColor(QPalette.Text, QColor(230, 240, 235))
        palette.setColor(QPalette.Button, QColor(40, 80, 60))
        palette.setColor(QPalette.ButtonText, QColor(230, 240, 235))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        return palette

    def get_stylesheet(self) -> str:
        return """QGroupBox {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #81C784;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #888888;
            }
            QTabWidget::pane {
                border: 2px solid #4CAF50;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2D5A3D;
                color: #E6F0EB;
                padding: 8px 20px;
                margin: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #388E3C;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2D5A3D;
                border: 1px solid #4CAF50;
                border-radius: 3px;
                padding: 4px;
                color: #E6F0EB;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #81C784;
            }
            QTextEdit {
                background-color: #1B3A28;
                border: 1px solid #4CAF50;
                border-radius: 3px;
                color: #E6F0EB;
            }
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                background-color: #2D5A3D;
            }
            QProgressBar::chunk {
                background-color: #81C784;
                border-radius: 3px;
            }
        """


class LightTheme(OrcaTheme):
    """Light theme - Clean bright color scheme."""

    def get_palette(self) -> QPalette:
        palette = QPalette()
        
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(250, 250, 250))
        palette.setColor(QPalette.ToolTipBase, QColor(33, 33, 33))
        palette.setColor(QPalette.ToolTipText, QColor(245, 245, 245))
        palette.setColor(QPalette.Text, QColor(33, 33, 33))
        palette.setColor(QPalette.Button, QColor(230, 230, 230))
        palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        return palette

    def get_stylesheet(self) -> str:
        return """QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #1565C0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #999999;
            }
            QTabWidget::pane {
                border: 2px solid #2196F3;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #EEEEEE;
                color: #333333;
                padding: 8px 20px;
                margin: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #E3F2FD;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #BDBDBD;
                border-radius: 3px;
                padding: 4px;
                color: #333333;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #2196F3;
            }
            QTextEdit {
                background-color: #FAFAFA;
                border: 1px solid #BDBDBD;
                border-radius: 3px;
                color: #333333;
            }
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
                background-color: #EEEEEE;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """