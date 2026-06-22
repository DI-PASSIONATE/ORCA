import inspect
import json
from typing import Type

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QCheckBox, QLabel, QLineEdit, 
    QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox
)

from orca.pipeline.pipeline_stage import PipelineStage

class StageConfigWidget(QWidget):
    """
    This widget represents a single PipelineStage configuration panel.
    """
    def __init__(self, stage_class: Type[PipelineStage], parent=None):
        super().__init__(parent)
        self.stage_class = stage_class
        self.parameter_inputs = {}
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.group_box = QGroupBox(self.stage_class.__name__)
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)
        
        form_layout = QFormLayout()
        self.group_box.setLayout(form_layout)
        
        # Introspect __init__
        sig = inspect.signature(self.stage_class.__init__)
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
                
            label = QLabel(name)
            input_widget = self.create_input_widget(param)
            
            self.parameter_inputs[name] = {"widget": input_widget, "type": param.annotation}
            form_layout.addRow(label, input_widget)
            
            # Set default value if available
            if param.default is not param.empty:
                self.set_widget_value(input_widget, param.default)
            
        layout.addWidget(self.group_box)
        
    def create_input_widget(self, param: inspect.Parameter):
        annotation = param.annotation
        
        # Handle simple types
        if annotation is int:
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            return widget
        elif annotation is float:
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            return widget
        elif annotation is bool:
            return QCheckBox()
        else:
            # Fallback for str, complex types, or unannotated
            return QLineEdit()

    def set_widget_value(self, widget, value):
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QLineEdit):
            if isinstance(value, (dict, list)):
                widget.setText(json.dumps(value))
            else:
                widget.setText(str(value))
                
    def get_widget_value(self, widget, annotation):
        if isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QLineEdit):
            text = widget.text()
            if annotation is int:
                return int(text)
            elif annotation is float:
                return float(text)
            elif annotation is bool:
                return text.lower() == "true"
            elif annotation is str:
                return text
            # Try to parse JSON for lists/dicts if it looks like one
            if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
                 try:
                     return json.loads(text)
                 except:
                     return text
            return text
        return None

    def is_enabled(self):
        return self.group_box.isChecked()
        
    def get_instance(self) -> PipelineStage | None:
        if not self.is_enabled():
            return None
            
        kwargs = {}
        for name, info in self.parameter_inputs.items():
            widget = info["widget"]
            annotation = info["type"]
            kwargs[name] = self.get_widget_value(widget, annotation)
            
        return self.stage_class(**kwargs)

