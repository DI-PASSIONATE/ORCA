# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `BaseGeometry.windings()`: the ports of each winding, written to the ONNX `windings`
  metadata. `InductorOcta` declares `[[1, 2, 3]]`,
  `TransformerOcta` `[[1, 2, 5], [3, 4, 6]]`.
- The ONNX `touchstone_type` metadata names the Palace result variant the model was trained on
  (`normal`, `dc`, `deembedded` or `dc_deembedded`), so COBRA's EM fine-tuning can use the same
  one.
- `input_parameter_ranges` gives the `step` of integer parameters (e.g. turns), whose geometry
  exists only on that grid.

## [2.2.0] - 2026-10-16

### Added

- `simulate_geometry(..., result_type=...)` returns the given Touchstone variant (for example
  `"deembedded"`, without the extrapolated 0 Hz point) instead of the most corrected one, and
  falls back to the raw result when that variant was not written.

## [2.1.1] - 2026-10-14

### Changed

- `TransformerOcta` accepts a wider parameter range: windings down to 0.55 (was 0.75) of their
  partner's diameter and a `relative_displacement` up to 0.3 (was 0.2). The ONNX
  `input_parameter_ranges` and `input_constraints` of newly trained models follow.
- `examples/main.py` runs the whole pipeline again, with the new training loss options.

### Fixed

- Port de-embedding cascaded the negative port inductances of all ports at port 1 and none at
  the others. Each port now gets its own. This changes the `_deembedded` and `_dc_deembedded`
  results of every multi-port device; re-run the merge, or the simulations, and retrain models
  trained on data simulated with 2.1.0 or earlier.
- The self-resonance frequency (`srf_f`) is no longer taken from sign flips of the reactance
  just above DC, where it is numerical noise around zero.

## [2.1.0] - 2026-10-06

### Added

- `orca.simulate_geometry(geometry, params, output_dir, name, ...)` draws, meshes and simulates
  one sample with Palace, the way the pipeline does, and returns its network; it raises
  `orca.SimulationError` when that fails. COBRA's EM fine-tuning uses it.
- `orca.simulation.simulate.patch_palace_config`, the Palace config overrides of
  `PalaceSimulator`, as a public function.
- `BaseGeometry.simulation_ports(params)` changes port fields (such as `to_layername`) per
  sample; the port numbers and order stay those of the simconfig.
- `BaseGeometry.electrical_parameters(ntwk)` and `absolute_error_parameters`: each geometry
  declares the figures of merit `ModelTester` reports and plots. `orca.utils.postprocessing`
  provides `inductor_parameters`, `transformer_parameters` and `plot_electrical_parameters`.
- `DRCChecker(check_ports=True)` checks that every port marker touches the metals its port
  connects, and always drops layouts with an open port.
- `GDSConverter` checks every new mesh for flat, inverted or corrupt elements and leaves those
  samples out (`min_element_quality`, `warn_element_quality`); the worst element quality of
  each mesh goes to `<name>_mesh_report.csv`.
- `ModelTrainer` loss options: `admittance_weight` (relative error of the admittance matrix),
  `passivity_weight` (penalty on non-passive predictions) and `above_srf_weight` (weight of the
  points above the first self-resonance). `orca.SParameterLoss` is the loss behind them.
- Simconfigs take extra frequencies in `fpoint`, and `fstart` may be 0 for a DC point.
- Slurm example scripts split simulation into a prepare job (GDS, DRC, conversion) and a
  simulate job (`examples/slurm_runs/submit.sh`).

### Changed

- `TransformerOcta` and `InductorOcta` share one stackup (`SG13G2_FEM_200um_passi3D`), the
  ground ring width and port definitions on the outer edge of the ground ring. The inductor
  sweep adds points near DC, and multi-turn inductors join their crossovers with at least two
  vias. Models trained on the old presets need retraining.
- The DC point is extrapolated from the first simulated frequencies only, instead of
  resampling the whole network onto an even grid.
- `GDSConverter(timeout=...)` defaults to 180 s (was 60 s).
- **Breaking:** `orca.utils.postprocessing` no longer has `calculate_electrical_parameters`,
  `plot_rfic_transformer_metrics` or `to_mixed_mode`; use the functions listed above.
- **Breaking:** the inductor cell's `ground_style` argument is removed; it always draws a ring.

## [2.0.0] - 2026-10-03

### Added

- Interrupted runs resume: `GDSGenerator`, `GDSConverter` and `PalaceSimulator` record each
  finished sample right away and, started again, only produce the missing ones. Each takes
  `overwrite=True` to start from scratch instead.
- `ORCA.run(seed=...)`, also a **Seed** field in the GUI: one seed for the whole run (parameter
  draws, data splits, tuning, weight initialisation).
- `RangeParameter` and `ChoiceParameter` declare a geometry's parameters, with a step, an
  integer type and uniform or log sampling per parameter.
- Sobol' (now the default) and Latin hypercube sampling, with a share of the samples on the
  boundary of the parameter box (`boundary_fraction`). `kept_on_rejection` keeps chosen
  parameters evenly distributed when infeasible draws are rejected.
- `GDSGenerator` plots the sampled parameters to `geometries/<name>_coverage.png`.
- `ModelTrainer` options: `val_frac`, `tuning_timeout`, `max_epochs`, `tuning_max_epochs`,
  `batch_sizes`, `regularization`, `lr_schedule`, `warmup_epochs`, `grad_clip_norm` and
  `allow_tf32`. `n_fold_cv=1` tunes on the validation split, without cross-validation.
- The tuned hyperparameters are saved to `models/<name>_hyperparameters.json`, and
  `ModelTrainer(hyperparameters=...)` accepts the path of such a file.
- `ModelTester` reports error percentiles, the worst geometry and the error per frequency band
  (`n_frequency_bands`), and writes per-geometry errors and an error-vs-frequency plot to
  `models/`.
- `StackupXML` enum for the stackups shipped with the presets.
- Slurm example scripts for training on a GPU cluster (`examples/slurm_runs/train.py`).

### Changed

- **Breaking:** `InputParameterIterator` takes `RangeParameter` / `ChoiceParameter` objects
  instead of value lists as keyword arguments.
- **Breaking:** `GDSGenerator(seed=...)` is removed; use `ORCA.run(seed=...)`.
- **Breaking:** Presets moved into one folder per device; import them as
  `from orca.geometry.presets import InductorOcta, TransformerOcta`.
- **Breaking:** The presets use gds2palace's current IHP stackups (`SG13G2_FEM_200um`,
  `SG13G2_FEM_200um_passi3D`); `SG13G2_nosub.xml` and `SG13G2_200um.xml` are removed.
- **Breaking:** `TransformerOcta` takes `relative_displacement` (a share of the mean winding
  diameter) instead of `center_displacement`, and its layout was reworked.
- `InductorOcta` has a centre-tap port, a ring ground on Metal5, allows a single turn, and uses
  more accurate simulation settings. Models trained on the old presets need retraining.
- Data is split by geometry, never by frequency point, and the split is saved to
  `models/<name>_split.csv` for `ModelTester`.
- Faster training: each Touchstone file is parsed once per run, and batches are drawn from
  tensors kept on the GPU.
- A simulation that Slurm refuses to launch is retried after a delay before it counts as failed.
- The GUI collapses unticked stages and only asks for confirmation when a stage will overwrite
  earlier results.
- Requires gds2palace 0.8.0 or newer.
- The Hugging Face tag for shared models is now `orca-rfic` (was `orca-surrogate`).

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
