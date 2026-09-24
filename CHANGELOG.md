# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2026-09-24

### Added

- ORCA can be installed from PyPI: `pip install "orca-rfic[train]"`. The import package and
  the `orca` command keep their names.
- `DRCChecker` stage: snaps GDS files to the manufacturing grid, checks them against the SG13G2
  design rules, and leaves out layouts that fail.
- `BaseGeometry.is_feasible()`: rejects parameter combinations that can't be built; the
  sampler draws new ones in their place.
- `BaseGeometry.feasibility_constraints()`: the same rules as expressions, exported to the
  ONNX metadata (`input_constraints`) for COBRA.

### Changed

- The GUI matches COBRA's look and has a light/dark theme switch.

## [1.4.0] - 2026-09-16

### Added

- `train` extra: install with `pip install -e ".[train]"` to train, export and test models.
  GDS generation, conversion and simulation work without it.
- `GDSConverter(timeout=...)`: skips conversions that take longer than the timeout.
- `GDSGenerator(seed=...)` for reproducible sampling.

### Changed

- **Breaking:** geometries implement `create_dataset()` instead of setting a `dataset` field.
- Supports Python 3.11–3.13.

## [1.3.0] - 2026-09-14

### Added

- `PalaceSimulator(launcher="local" | "slurm", num_parallel_sims=..., bind=...)`: runs several
  simulations at once, locally or across a Slurm allocation, each on its own node, socket or
  NUMA domain.
- `PalaceSimulator(save_log=True)` keeps each simulation's Palace output.

### Changed

- **Breaking:** `PalaceSimulator(num_parallel_palace_sims=...)` is now `num_parallel_sims`.
- The IHP PDK is no longer required.
- A script that starts the pipeline must do so inside `if __name__ == "__main__":`.

## [1.2.0] - 2026-09-11

### Added

- Swappable model architectures (`OrcaModel`, `register_model`), output representations
  (`FlatReImCodec`, `UpperTriangleReImCodec`) and input expansions (`BasisExpansion`,
  `ChebyshevBasis`).
- `ORCA.run(result_dir=..., result_csv=...)`: train on results that were simulated earlier.
- `ModelTester` also tests an exported ONNX model.
- Physical guarantees of a model are written to the ONNX metadata (`physics_guarantees`).

### Changed

- **Breaking:** stages receive and return a `PipelineContext` instead of a `dict`.
- **Breaking:** the model is chosen with `ModelTrainer(model=..., basis=...)` instead of on the
  geometry.
- **Breaking:** `FeatureTransform` is replaced by `BasisExpansion`.

### Fixed

- Training several geometries in one process no longer mixes up their normalization.

## [1.1.0] - 2026-09-09

### Added

- `InductorOcta` preset: a 2-port octagonal spiral inductor.
- Running on HPC clusters with Slurm; see `examples/slurm_runs/`.
- `ORCA.run(force_overwrite=True)` for runs without a confirmation prompt.

### Changed

- **Breaking:** `ORCA.run(cpu_cores=...)` is now `num_processes`.
- PyTorch is no longer installed automatically.

## [1.0.1] - 2026-06-22

### Changed

- Improved documentation.

## [1.0.0] - 2026-06-10

### Added

- Added CONTRIBUTING.md and CHANGELOG.md

### Changed

- Updated README to explain how to properly use ORCA
