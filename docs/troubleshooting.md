---
title: Troubleshooting
description: >-
  Fixes for common ORCA problems: orca command not found, failing Palace EM
  simulations, gds2palace GDS conversion errors and ONNX export issues.
---

# Troubleshooting

## `orca` command not found

Ensure your virtual environment is activated and reinstall with `pip install -e .` (or `uv pip install -e .`).

## Palace simulations fail

Verify Palace is installed and available in your `PATH`, or adjust the `palace_executable` argument of `PalaceSimulator` (e.g. `apptainer exec palace.sif palace` for a containerised install). Pass `save_log=True` to keep the full Palace output of each simulation in `palace.log` in its simulation folder.

## GDS conversion fails

Verify that [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2) is installed and that the stackup XML matches your technology. See [Custom Classes](custom_class.md#stackup-xml-file) for the stackup format.

## `ModuleNotFoundError` for torch, sklearn, optuna or onnx

The model-training packages are an optional extra. `orca.ModelTrainer`, `orca.OnnxExporter` and `orca.ModelTester` need them; install with `pip install -e ".[train]"` (see [Setup](setup.md)). The GDS and simulation stages work without them.

## Still stuck?

Please [open an issue](https://github.com/DI-PASSIONATE/ORCA/issues) on GitHub with the log output and a description of your setup.
