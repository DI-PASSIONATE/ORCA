# ORCA: An AI-Assisted Surrogate Modelling Pipeline for RFIC Passives

<div style="display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;">
	<img src="logo.png" alt="ORCA logo" width="300"/>
	<p style="margin:0; flex:1; min-width:240px;">
		ORCA (Open RF Integrated Circuit Automation) builds surrogate models of RF integrated circuit passives by combining parametric GDS layout generation, full-wave electromagnetic simulation, and neural network training and export in one automated pipeline.
	</p>
</div>


<div style="display:grid; grid-template-columns: minmax(260px, 1.2fr) minmax(180px, 0.8fr); gap:1.2rem; align-items:start;">
	<div>
		<p style="margin:0 0 .35rem 0; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; opacity:.8;">Authors</p>
		<p style="margin:0; line-height:1.55;">
			<a href="https://orcid.org/0009-0002-6315-1538">Gianluca Simone</a>,
			<a href="https://orcid.org/0009-0003-6526-5464">David Lurz</a>,
			<a>Martin Grund</a>,
			<a href="https://orcid.org/0009-0007-5891-7582">Fabian Schneider</a>,
			<a href="https://orcid.org/0000-0002-8422-5391">Michael Loose</a>,
			<a href="https://orcid.org/0000-0002-9600-2988">Sascha Breun</a>,
			<a href="https://orcid.org/0009-0002-7827-7205">Manuel Koch</a>,
			<a href="https://scholar.google.com/citations?user=74ugHPcAAAAJ&hl=de">Robert Weigel</a>,
			<a href="https://orcid.org/0000-0002-2777-4722">Norman Franchi</a>
		</p>
	</div>
	<div>
		<p style="margin:0 0 .35rem 0; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; opacity:.8;">Published</p>
		<p style="margin:0;">SBCCI 2026 (conference contribution)</p>
	</div>
</div>


## Main Capabilities

- Parametric GDS layout generation via [gdsfactory](https://github.com/gdsfactory/gdsfactory).
- Full-wave electromagnetic simulation via [Palace](https://github.com/awslabs/palace).
- Automatic dataset construction from simulation results.
- Neural network training (PyTorch) with feature engineering pipelines.
- ONNX export for portable, simulator-ready surrogate models.
- GUI workflow and script-first workflow.
- Pre-trained models and geometry examples included.

## Start Here

- New users: go to **Getting Started -> Installation**.
- First successful run: **Getting Started -> Quickstart**.
- Bring your own geometry: **User Guide -> Custom Classes**.
- API details: see the source and docstrings in `src/orca/`.

## Core Concepts

ORCA automates the full loop from geometry to a trained, exported surrogate model.

### Pipeline Workflow

```mermaid
flowchart LR
		A[Geometry Class] --> B[GDSGenerator]
		B --> C[GDSConverter]
		C --> D[PalaceSimulator]
		D --> E[S-Parameter Dataset]
		E --> F[ModelTrainer]
		F --> G[OnnxExporter]
		G --> H[ModelTester]
		H --> I[.onnx Surrogate]
```

### Pipeline Stages

| Stage | Purpose |
|---|---|
| `GDSGenerator` | Samples geometry parameters and writes GDS layout files |
| `GDSConverter` | Converts GDS files to Palace-compatible mesh inputs |
| `PalaceSimulator` | Runs full-wave EM simulations and stores Touchstone results |
| `ModelTrainer` | Trains a PyTorch neural network on the simulation dataset |
| `OnnxExporter` | Exports the trained model to portable ONNX format |
| `ModelTester` | Evaluates prediction accuracy against held-out simulation data |

### Geometry Class

A geometry class defines everything ORCA needs to sample and simulate a component:

- **Parameter ranges** via `InputParameterIterator` (sampling strategy, min/max per parameter).
- **GDS generation** via `create_gds_file(name, output_path, params)`.
- **Feature engineering** via `FeatureTransformPipeline` (ratios, Chebyshev features, etc.).
- **Dataset type** (e.g. `GeoToSParamDatasetSingleFrequency`).
- **Model factory** via `get_model(hyperparameters)` — called by `ModelTrainer` during Optuna tuning.
- **Hyperparameter search space** via `get_hyperparameter_search_space()` — Optuna distributions per tunable parameter.

## Inputs and Outputs

=== "Inputs"

		- A geometry class (`BaseGeometry` subclass) with parameter definitions.
		- A stackup XML file describing the physical layer stack.
		- A Palace simulation configuration file (`.simcfg`).
		- A Palace executable (local or containerised via Apptainer).

=== "Outputs"

		- Touchstone (`.sNp`) S-parameter files per simulated geometry variant.
		- Trained PyTorch model checkpoint.
		- Exported ONNX surrogate model (`.onnx`).
		- Full run context stored under `output/<geometry_name>/`.

## Related Projects

- COBRA consumes ORCA surrogate models for circuit-level RFIC optimization.
- Palace is used as the EM solver backend.

---

If you use ORCA in research, see citation information in the repository README.