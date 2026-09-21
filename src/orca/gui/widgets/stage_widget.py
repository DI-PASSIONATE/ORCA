import inspect
import json
import types
import typing

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from orca.gui.help_texts import parameter_tooltips, tooltip
from orca.gui.theme import manager as theme_manager
from orca.pipeline.pipeline_stage import PipelineStage


class StageConfigWidget(QWidget):
    """
    This widget represents a single PipelineStage configuration panel.
    """
    def __init__(self, stage_class: type[PipelineStage], parent=None):
        super().__init__(parent)
        self.stage_class = stage_class
        self.parameter_inputs = {}

        self.init_ui()

    def init_ui(self):
        tokens = theme_manager().tokens
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.group_box = QGroupBox(self.stage_class.__name__)
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)
        self.group_box.setToolTip(tooltip("stage_group"))

        form_layout = QFormLayout()
        form_layout.setSpacing(tokens.space_2)
        form_layout.setContentsMargins(
            tokens.space_3, tokens.space_2, tokens.space_3, tokens.space_3
        )
        self.group_box.setLayout(form_layout)

        # Introspect __init__; the Args docstring supplies the tooltips
        sig = inspect.signature(self.stage_class.__init__)
        tips = parameter_tooltips(self.stage_class)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            label = QLabel(name.replace("_", " ").capitalize())
            input_widget = self.create_input_widget(param)
            if name in tips:
                label.setToolTip(tips[name])
                input_widget.setToolTip(tips[name])

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
        if annotation is float:
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            return widget
        if annotation is bool:
            return QCheckBox()
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
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QLineEdit):
            text = widget.text()
            # Optional parameters (`int | None`) are edited as text; an empty field or
            # the literal "None" means None, anything else is parsed as the other type.
            if typing.get_origin(annotation) in (types.UnionType, typing.Union):
                members = [m for m in typing.get_args(annotation) if m is not type(None)]
                if text.strip() in ("", "None"):
                    return None
                if len(members) == 1:
                    annotation = members[0]
            if annotation is int:
                return int(text)
            if annotation is float:
                return float(text)
            if annotation is bool:
                return text.lower() == "true"
            if annotation is str:
                return text
            # Try to parse JSON for lists/dicts if it looks like one
            if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
                 try:
                     return json.loads(text)
                 except json.JSONDecodeError:
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

