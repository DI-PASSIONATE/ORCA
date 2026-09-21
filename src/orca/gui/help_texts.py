"""Tooltips of the ORCA GUI, in one place so wording stays consistent.

Static controls look their text up by key with :func:`tooltip`; the stage
parameter forms, which are derived from the stage constructors, take theirs
from the ``Args:`` section of the constructor docstring via
:func:`parameter_tooltips`, so a stage documents its options once.
"""

from __future__ import annotations

import inspect
import re

TOOLTIPS = {
    "theme_btn": "Appearance: {mode} ({theme}). Click to switch between system, light and dark.",
    "run_btn": "Run the enabled stages in order on the selected geometry.",
    "geometry_combo": "Built-in geometry presets. Loading a custom file replaces the selection.",
    "geometry_file_btn": "Browse for a Python file that defines a BaseGeometry subclass.",
    "geometry_name_edit": (
        "Name of the run. Output goes to output/<name>/; leave the preset name to keep "
        "its default."
    ),
    "stage_group": (
        "Tick to include this stage in the run. A stage whose inputs are missing "
        "reports which earlier stage was skipped."
    ),
    "log_output": "Messages from the pipeline; the same text ORCA prints on the console.",
}

#: ``name (type): text`` or ``name: text`` at the start of an Args entry.
_ARG_LINE = re.compile(r"^(?P<name>\w+)(?:\s*\([^)]*\))?:\s*(?P<text>.*)$")


def tooltip(key: str) -> str:
    return TOOLTIPS.get(key, "")


def parameter_tooltips(cls: type) -> dict[str, str]:
    """Per-parameter descriptions from the ``Args:`` section of ``cls.__init__``'s docstring.

    Continuation lines are joined with spaces; parameters without an entry are
    absent from the result.
    """
    doc = inspect.getdoc(cls.__init__)
    if not doc:
        return {}
    tips: dict[str, str] = {}
    current: str | None = None
    in_args = False
    for raw in doc.splitlines():
        line = raw.strip()
        if not in_args:
            in_args = line == "Args:"
            continue
        if not line or (raw and not raw[0].isspace()):
            # A blank line or a new section heading ends the Args block.
            break
        match = _ARG_LINE.match(line)
        if match and not raw.startswith("        "):
            current = match.group("name")
            tips[current] = match.group("text")
        elif current is not None:
            tips[current] = f"{tips[current]} {line}".strip()
    return tips
