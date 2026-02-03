"""
Utility module for discovering classes that inherit from specific base classes.
"""

import importlib
import inspect
import os
from pathlib import Path
from typing import Type, Dict, Any

from orca.logger import logger


def discover_classes(
    base_class: Type,
    search_dir: str,
    module_prefix: str,
    extract_default_params: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Automatically discover and load classes that inherit from a specified base class.

    Args:
        base_class: The base class to search for (e.g., BaseGeometry, nn.Module)
        search_dir: Absolute path to the directory to search for Python files
        module_prefix: The Python module path prefix (e.g., "orca.geometry.presets")
        extract_default_params: Whether to extract default parameters from __init__

    Returns:
        Dictionary mapping display names to class information including:
        - class: The class object
        - default_params: Dictionary of default parameters (if extract_default_params=True)
        - module: Module name
        - class_name: Class name
    """
    classes = {}

    if not os.path.exists(search_dir):
        logger.warning(f"Search directory not found: {search_dir}")
        return classes

    # Iterate through Python files in the directory
    search_path = Path(search_dir)
    for py_file in search_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            # Import the module
            module_name = py_file.stem
            full_module_name = f"{module_prefix}.{module_name}"
            module = importlib.import_module(full_module_name)

            # Find classes that inherit from the base class
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and obj is not base_class
                ):
                    try:
                        # Build class information
                        class_info = {
                            "class": obj,
                            "module": module_name,
                            "class_name": name,
                        }

                        # Extract default parameters if requested
                        if extract_default_params:
                            sig = inspect.signature(obj.__init__)
                            default_params = {}
                            for param_name, param in sig.parameters.items():
                                if param_name in ("self", "params"):
                                    continue
                                if param.default != inspect.Parameter.empty:
                                    default_params[param_name] = param.default
                            class_info["default_params"] = default_params

                        # Use a descriptive display name
                        display_name = module_name
                        classes[display_name] = class_info

                        logger.info(
                            f"Discovered {base_class.__name__} subclass: {display_name}"
                        )

                    except Exception as e:
                        logger.warning(f"Could not process {name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load classes from {py_file.name}: {e}")

    if not classes:
        logger.warning(f"No {base_class.__name__} subclasses found in {search_dir}")

    return classes
