## Project Purpose

ORCA (Open RF Integrated Circuit Automation) builds neural-network surrogate
models of RFIC passives. A run is a pipeline: generate GDS layouts with
gdsfactory → convert them for Palace (`gds2palace`) → EM-simulate with Palace →
train a PyTorch model on the S-parameters → export it to ONNX → test it. The
ONNX models are consumed by COBRA (`../COBRA`), which reads the
`input_parameter_ranges` and `physics_guarantees` metadata ORCA writes.

## Repository Structure

- `src/orca/orca.py`: the `ORCA` runner; sorts stages by `index` and runs them
  over one `PipelineContext`. Output goes to `output/<geometry name>/`.
- `src/orca/pipeline/`: `PipelineStage` base, `PipelineContext`, and the stages
  in fixed order: `GDSGenerator` (0), `GDSConverter` (1), `PalaceSimulator` (2),
  `ModelTrainer` (4), `OnnxExporter` (5), `ModelTester` (6).
- `src/orca/geometry/`: `BaseGeometry` contract, `InputParameterIterator`,
  layer stackups, reusable cells, and presets (`inductor_octa`,
  `tf_octa_c_ports`) with their `.simcfg`/`.xml` package data.
- `src/orca/simulation/`: GDS→Palace conversion, Palace launchers (local,
  Apptainer, Slurm), and Touchstone result merging.
- `src/orca/training/`: models, datasets, output codecs, basis expansions,
  physics guarantees, trainer, tuner, ONNX wrapper, predictors.
- `src/orca/gui/`: PySide6 pipeline window and widgets; `orca` with no
  arguments launches it. There is no CLI beyond that.
- `examples/main.py`: the canonical pipeline script; `examples/slurm_runs/`:
  HPC job scripts. `docs/`: MkDocs site, deployed from `main` by CI.

Find the owner of a behaviour before editing. Geometry, simulation, training,
pipeline orchestration, and GUI stay in their own packages.

## Contracts to Preserve

- **Stages:** subclass `PipelineStage`, keep the fixed `index`, and implement
  `run(context, progress_callback) -> context`. Read only the context fields
  earlier stages own and write only your own; the ownership groups are
  documented in `pipeline/context.py`. A stage whose inputs are missing must
  say which stage was skipped, not fail on a `KeyError`.
- **Geometries:** a `BaseGeometry` subclass provides `name`, `stackup_xml`,
  `simconfig_filename`, `input_parameter_iterator`, `create_gds_file`, and
  `create_dataset`. Presets are examples of the contract; a change to the
  contract updates the presets and `docs/custom_class.md`.
- **Public API:** everything importable as `orca.X` is listed in
  `src/orca/__init__.py`. Anything that needs PyTorch goes in
  `_TRAINING_EXPORTS` and is imported lazily; `import orca`, `GDSGenerator`,
  `GDSConverter`, and `PalaceSimulator` must keep working in an install
  without the `train` extra (simulation-only HPC nodes). Keep torch imports
  local inside functions and stages for the same reason.
- **ONNX export:** the model file is what COBRA loads. Do not change the input
  names, the output codec, or the metadata keys without changing COBRA too.

## Python

- Use uv with the environment in `.venv/`: `uv sync --extra train --extra cpu`
  for development (CI uses the same); `uv sync` alone gives a simulation-only
  environment. Run tools as `uv run <command>` or via `.venv/bin/<command>`.
  Do not silently use a global interpreter.
- Follow the existing type-hint, dataclass, naming, and import style; Google
  docstrings; line length 100. RF symbol names (`S`, `Z`, `N`) keep their
  casing, and `os.path` is used throughout — do not migrate either in passing.
- Prefer extending the existing base classes (`PipelineStage`, `BaseGeometry`,
  `OrcaModel`, `OutputCodec`, `BasisExpansion`, `BaseDataset`) over parallel
  utility APIs. New models and bases go through `register_model` /
  `register_basis`.
- After changing Python files run `uv run ruff check` and
  `uv run ty check`; CI fails on either. Ruff runs with `select = ALL`
  and a curated ignore list in `pyproject.toml` — add to that list only with a
  one-line reason, as the existing entries do.
- There is no test suite yet (`tests/` is empty, pytest is configured). When
  adding tests, put them in `tests/`, add `pytest` to the `dev` group, and keep
  them free of Palace, torch training, and network access.

## Simulation and Training

- Never run Palace, GDS generation for thousands of samples, or model training
  in the foreground; they take minutes to hours. Use a background job or
  reduce `num_samples`/`epochs` for a smoke test.
- `default_process_count()` respects the Slurm/container affinity mask; use it
  instead of `os.cpu_count()` for MPI rank defaults.

## GUI

- Read `.claude/GUI_DESIGN.md` before editing `src/orca/gui/`. It defines the
  themes (Sandbank light / Deepwater dark), colour tokens, component rules,
  icon usage, and the both-themes check to run before finishing. The tokens are
  shared with COBRA; `../COBRA/src/cobra/gui/theme.py` is the reference
  implementation.
- The GUI derives each stage's form from the stage constructor's signature
  (`gui/widgets/stage_widget.py`); a new constructor argument appears there
  automatically, so give it a type hint and a default.

## Documentation

Update the closest page in `docs/` when a stage, geometry contract, install
step, or GUI behaviour changes (`pipeline.md`, `custom_class.md`, `setup.md`,
`running_orca.md`), and `examples/main.py` when the canonical pipeline call
changes. Keep README, docs, and code consistent; call out discrepancies.
