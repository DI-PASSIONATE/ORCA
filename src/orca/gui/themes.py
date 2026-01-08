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


class SageTheme(OrcaTheme):
    def get_palette(self) -> QPalette:
        palette = QPalette()
        
        # Sage/Navy palette
        # Very light sage: 235,244,221 | Sage: 144,171,139 | Deep sage: 90,120,99 | Navy: 59,73,83
        palette.setColor(QPalette.ColorRole.Window, QColor(59, 73, 83))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(23, 24, 22))
        palette.setColor(QPalette.ColorRole.Base, QColor(90, 120, 99))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(144, 171, 139))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(235, 244, 221))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(59, 73, 83))
        palette.setColor(QPalette.ColorRole.Text, QColor(235, 244, 221))
        palette.setColor(QPalette.ColorRole.Button, QColor(144, 171, 139))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(59, 73, 83))
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(144, 171, 139))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        return palette

    def get_stylesheet(self) -> str:
        return """QGroupBox {
                border: 2px solid #90AB8B; /* sage */
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #EBF4DD; /* very light sage text */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                color: #333333;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #90AB8B; /* sage */
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                color: #3B4953; /* navy text */
            }
            QPushButton:hover {
                background-color: #5A7863; /* deep sage */
            }
            QPushButton:pressed {
                background-color: #3B4953; /* navy */
                color: #EBF4DD; /* ensure contrast when pressed */
            }
            QPushButton:disabled {
                background-color: #5A7863; /* deep sage muted */
                color: #90AB8B; /* lighter muted text */
            }
            QTabWidget::pane {
                border: 2px solid #90AB8B; /* sage */
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #3B4953; /* navy */
                color: #EBF4DD; /* very light sage */
                padding: 8px 20px;
                margin: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #90AB8B; /* sage */
                color: #3B4953; /* navy text */
            }
            QTabBar::tab:hover {
                background-color: #5A7863; /* deep sage */
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3B4953; /* navy */
                border: 1px solid #90AB8B; /* sage */
                border-radius: 3px;
                padding: 4px;
                color: #EBF4DD; /* very light sage */
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #5A7863; /* deep sage */
            }
            QTextEdit {
                background-color: #3B4953; /* navy */
                border: 1px solid #90AB8B; /* sage */
                border-radius: 3px;
                color: #EBF4DD; /* very light sage */
            }
            QProgressBar {
                border: 2px solid #90AB8B; /* sage */
                border-radius: 5px;
                text-align: center;
                background-color: #3B4953; /* navy */
            }
            QProgressBar::chunk {
                background-color: #90AB8B; /* sage */
                border-radius: 3px;
            }
        """



class LightTheme(OrcaTheme):
    """Light theme - Clean bright color scheme."""

    def get_palette(self) -> QPalette:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(59, 73, 83))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(23, 24, 22))
        palette.setColor(QPalette.ColorRole.Base, QColor(90, 120, 99))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(144, 171, 139))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(235, 244, 221))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(59, 73, 83))
        palette.setColor(QPalette.ColorRole.Text, QColor(235, 244, 221))
        palette.setColor(QPalette.ColorRole.Button, QColor(144, 171, 139))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(59, 73, 83))
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(241, 109, 52))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
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
                color: #333333;
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