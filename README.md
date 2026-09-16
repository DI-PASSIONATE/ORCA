# ORCA — Open RF Integrated Circuit Automation

**An open-source EDA tool for AI-assisted RFIC design: a surrogate modelling pipeline for RF integrated circuit components — from parametric GDS layout to full-wave EM simulation to a trained ONNX model for COBRA.**

[![Documentation](https://img.shields.io/badge/docs-di--passionate.github.io%2FORCA-green?logo=materialformkdocs&logoColor=white)](https://di-passionate.github.io/ORCA/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](pyproject.toml)
[![Cite this repository](https://img.shields.io/badge/cite-CITATION.cff-lightgrey)](CITATION.cff)
[![GitHub stars](https://img.shields.io/github/stars/DI-PASSIONATE/ORCA?style=social)](https://github.com/DI-PASSIONATE/ORCA/stargazers)

©2026

Gianluca Simone\*, David Lurz\*, Martin Grund\*, Fabian Schneider°, Michael Loose\*, Sascha Breun\*, Manuel Koch\*, Robert Weigel\*, Norman Franchi\*

\* Institute for Intelligent Electronics and Systems (LITES), Friedrich-Alexander-Universität (FAU), Erlangen-Nürnberg, Germany

° Chair of Integrated Electronic Systems, Otto-von-Guericke-University Magdeburg, Germany

[Paper (Coming Soon)](#cite-this-work) | [Documentation](https://DI-PASSIONATE.github.io/ORCA/) | [BibTeX](#cite-this-work)

> [!NOTE]
> ORCA is still under active development. The current codebase is functional and can be used for experimentation, but we keep adding features, improving documentation, and refining the API. If you encounter any issues or have questions, please [open an issue](https://github.com/DI-PASSIONATE/ORCA/issues) or reach out.

**ORCA** is an open-source EDA tool for AI-assisted RFIC design: a pipeline for building neural network surrogate models of RF integrated circuit components from parametric layouts (the bundled examples are on-chip inductors and transformers). Instead of running a slow electromagnetic (EM) simulation for every candidate geometry during circuit design, ORCA runs the simulations once, learns the mapping from geometry parameters to S-parameters, and hands the result to circuit optimizers like [COBRA](https://github.com/DI-PASSIONATE/COBRA) as a portable ONNX model.
It combines:

- parametric GDS layout generation (via [gdsfactory](https://github.com/gdsfactory/gdsfactory)),
- full-wave electromagnetic simulation (via [Palace](https://github.com/awslabs/palace)),
- and machine learning model training and export (PyTorch → ONNX).

Given a geometry class with configurable parameters, ORCA automatically:

1. generates thousands of differently parameterized GDS layout variants,
2. converts them to simulation meshes and runs full-wave EM simulations,
3. trains a neural network to predict S-parameters from geometry inputs,
4. exports the trained model to a portable ONNX file,
5. and tests the model against held-out simulation data.

The resulting ONNX model can then be loaded by [COBRA](https://github.com/DI-PASSIONATE/COBRA) for fast circuit-level optimization — no EM simulation required at optimization time. Created models can easily be shared via Hugging Face Hub for others to use in their own design flows and reduce redundant EM simulations across the community.

![ORCA pipeline overview: parametric GDS generation, Palace EM simulation, PyTorch training and ONNX export of an RFIC component surrogate model](docs/orca.png)

## Key Features

- **Parametric layout generation** — sample thousands of GDS variants of an RFIC component from a small Python geometry class.
- **Open-source full-wave EM simulation** — finite-element S-parameter extraction with [Palace](https://github.com/awslabs/palace); runs on a laptop, a multi-socket workstation or a Slurm HPC cluster.
- **Machine learning surrogate models** — PyTorch models with automatic normalization and Optuna hyperparameter tuning.
- **Portable ONNX export** — the trained model runs with `onnxruntime` only; no PyTorch needed at inference time.
- **Built-in validation** — held-out test set evaluation of the exported model.
- **GUI and scripting workflows** — click through the pipeline or drive it from Python; remote execution on OpenStack.
- **Model sharing** — publish surrogates on the Hugging Face Hub (`orca-surrogate` tag) for the whole community to reuse.
- **Technology-agnostic** — bring your own layer stackup XML; no PDK dependency.

## How ORCA Fits with COBRA

ORCA is the model-building side of the flow.

- **ORCA** takes a geometry class, runs EM simulations, and produces a trained surrogate model (e.g. `tf_octa_c_ports.onnx`).
- [COBRA](https://github.com/DI-PASSIONATE/COBRA) loads that ONNX model and uses it to predict S-parameters during optimization loops, without re-running EM simulations.
- When COBRA's optional EM fine-tuning is enabled, it calls back into ORCA's geometry classes to regenerate GDS layouts for verification.

In short: **ORCA builds the model, COBRA uses it** to optimize circuits quickly and can verify/refine with real EM simulations.

## Installation

### Requirements

- Python 3.11+
- [Palace](https://awslabs.github.io/palace/stable/) (for running EM simulations)

Install Palace separately by following [the Palace installation instructions](https://awslabs.github.io/palace/stable/install/index.html). Recommendations: apptainer for local installation / testing, spack for HPC clusters.


### Option A: Using `uv` (recommended)

1. Clone the repository:

```bash
git clone https://github.com/DI-PASSIONATE/ORCA
cd ORCA
```

2. Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install a supported Python version:

```bash
uv python install 3.13
```

4. Create and activate a virtual environment:

```bash
uv venv --python 3.13
source .venv/bin/activate
```

5. Install ORCA in editable mode. The `train` extra adds PyTorch and the other model-training dependencies; leave it out for a simulation-only install (GDS generation, conversion and Palace simulation), e.g. on an HPC cluster:

```bash
uv pip install -e ".[train]"   # full pipeline, PyTorch from PyPI
uv pip install -e .            # simulation only, no PyTorch
```

   To pick a specific PyTorch build, use `uv sync` with one of the build selectors defined in `pyproject.toml` instead:

```bash
uv sync --extra train --extra cpu     # CPU-only PyTorch wheels
uv sync --extra train --extra cu130   # CUDA 13.0 wheels (driver >= 580)
uv sync --extra train --extra cu126   # CUDA 12.6 wheels for older drivers
```

### Option B: Using standard `venv` + `pip`

1. Clone the repository:

```bash
git clone https://github.com/DI-PASSIONATE/ORCA
cd ORCA
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install ORCA. The `train` extra adds PyTorch and the other model-training dependencies; leave it out for a simulation-only install:

```bash
pip install -U pip
pip install -e ".[train]"   # full pipeline
pip install -e .            # simulation only, no PyTorch
```

   pip installs the PyPI build of PyTorch (CUDA-bundled on Linux). For a CPU-only or a specific CUDA build, install torch first with the command from [PyTorch.org](https://pytorch.org/get-started/locally/), then run the `pip install -e ".[train]"` line above; pip keeps the version already installed.

## Running ORCA

ORCA supports three usage modes.

### 1. GUI mode

After installation, start the GUI with:

```bash
orca
```

The GUI lets you:

- select a geometry preset or load a custom geometry class,
- configure pipeline stages and parameters,
- monitor simulation and training progress in real time,
- inspect and test the trained model.

### 2. Python script mode

For direct integration into scripts or automated workflows:

```python
import orca
from orca import ORCA
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

geometry = TransformerOcta()

orca_instance = ORCA(
    [
        orca.GDSGenerator(num_samples=1000),
        orca.GDSConverter(),
        orca.PalaceSimulator(palace_executable="palace"),
        orca.ModelTrainer(),
        orca.OnnxExporter(),
        orca.ModelTester(),
    ]
)

if __name__ == "__main__":
    orca_instance.run(geometry=geometry, num_processes=16)
```

Wrap the call in `if __name__ == "__main__":` (or a `main()` function) as shown: ORCA runs GDS generation and conversion in worker processes, and on platforms whose default start method is `spawn` (macOS, Windows) every worker re-imports your script. Without the guard each worker would start its own pipeline.

This generates 1000 parameterized layout variants, runs EM simulations, trains a model, exports it to ONNX, and evaluates its accuracy.
You can omit any stage (e.g. skip `GDSGenerator` and `GDSConverter` if simulation data already exists).

### 3. OpenStack remote execution

For large-scale simulation runs, we provide an OpenStack VM image and a CLI controller in the [ORCA-OpenStack repository](https://github.com/DI-PASSIONATE/ORCA-OpenStack). This lets you launch simulation and training jobs on a remote server without managing the environment manually.

## Pipeline Stages

| Stage | Class | Description |
|-------|-------|-------------|
| GDS generation | `GDSGenerator` | Creates parameterized GDS layout files from a geometry class |
| GDS conversion | `GDSConverter` | Converts GDS files to Palace-compatible simulation meshes using gds2palace |
| EM simulation | `PalaceSimulator` | Runs full-wave EM simulations in Palace and stores results as Touchstone files |
| Model training | `ModelTrainer` | Trains a PyTorch MLP to map geometry parameters + frequency to S-parameters |
| ONNX export | `OnnxExporter` | Exports the trained model to a portable ONNX file for use in COBRA |
| Model testing | `ModelTester` | Evaluates prediction accuracy on held-out simulation data |

Each stage reads from and writes to a shared context dictionary, so stages can be run independently or composed freely.

## How ORCA Works Internally

ORCA runs a linear pipeline. Each stage receives a context dictionary and adds its outputs for the next stage.

```
┌──────────────────────────────────────────────────────────────┐
│                        ORCA pipeline                         │
│                                                              │
│  ┌──────────────┐   GDS files    ┌──────────────────┐        │
│  │ GDSGenerator │───────────────▶│   GDSConverter   │        │
│  │              │                │ (gds2palace mesh) │        │
│  └──────────────┘                └────────┬─────────┘        │
│                                           │ mesh files        │
│                                           ▼                  │
│                                  ┌──────────────────┐        │
│                                  │ PalaceSimulator  │        │
│                                  │ (full-wave EM)   │        │
│                                  └────────┬─────────┘        │
│                                           │ Touchstone .sNp  │
│                                           ▼                  │
│                                  ┌──────────────────┐        │
│                                  │  ModelTrainer    │        │
│                                  │ (PyTorch MLP)    │        │
│                                  └────────┬─────────┘        │
│                                           │ trained model    │
│                                           ▼                  │
│                                  ┌──────────────────┐        │
│                                  │  OnnxExporter    │────────▶ .onnx (→ COBRA)
│                                  └────────┬─────────┘        │
│                                           │                  │
│                                           ▼                  │
│                                  ┌──────────────────┐        │
│                                  │   ModelTester    │        │
│                                  └──────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### Stage 1 — GDS generation (`GDSGenerator`)

The geometry class's `input_parameter_iterator` samples parameter combinations (randomly or on a grid). For each combination, `create_gds_file()` is called to produce a GDS layout file. The number of samples is set by `num_samples`; `seed` makes the `"random"` picking strategy reproducible.

### Stage 2 — GDS conversion (`GDSConverter`)

Each GDS file is converted to a Palace-ready simulation setup using [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2). The geometry's `stackup_xml` defines the physical layer stackup and material properties; the `simconfig_filename` defines the simulation parameters (port positions, frequency sweep, mesh settings).

### Stage 3 — EM simulation (`PalaceSimulator`)

Palace runs a full-wave finite-element EM simulation for each layout variant and writes the S-parameters to a Touchstone file (`.sNp`). Simulations are distributed across available CPU cores. The `palace_executable` argument can point to a local binary or a container invocation (e.g. `apptainer exec palace.sif palace`).

Several simulations can run at once, each already parallelized internally with MPI (`num_processes` ranks per simulation). `bind` chooses what one simulation gets — a whole `"node"`, one `"socket"` or one `"numa"` domain — and `num_parallel_sims=0` uses all such slots:

- `PalaceSimulator(num_parallel_sims=0, bind="numa")` runs one simulation per NUMA domain of the current machine, each pinned with `numactl` (e.g. 2 or 4 at once on a multi-socket workstation).
- `PalaceSimulator(launcher="slurm", num_parallel_sims=0, bind="numa")` does the same on every node of a Slurm allocation (`sbatch --nodes=N`), each simulation launched as an `srun` job step pinned to its node and cores. See [examples/slurm_runs](examples/slurm_runs) for a job script.

Palace is memory-bandwidth bound, so several smaller simulations confined to their own NUMA domain usually give a higher throughput than one simulation spread over a whole node — as long as one simulation fits into a domain's memory (use `bind="socket"` otherwise). The layout is derived from the machine or allocation at runtime, so `num_parallel_sims` and `num_processes` are capped to what is actually available. Pass `save_log=True` to keep each simulation's full Palace output in `palace.log` in its simulation folder (off by default, Palace prints a lot); failures are reported either way.

### Stage 4 — Model training (`ModelTrainer`)

A PyTorch MLP is trained on the simulation data. Inputs are geometry parameters and frequency; outputs are the real and imaginary parts of each S-parameter entry. Normalization is defined in the geometry's dataset and applied automatically. An optional basis expansion of the inputs — for example a Chebyshev expansion of frequency — is chosen on the stage itself with `ModelTrainer(basis="chebyshev")`; it lives inside the model, so it is tuned with it and exported into the ONNX graph. Hyperparameters such as learning rate, batch size, and network depth can be passed to `ModelTrainer`.

### Stage 5 — ONNX export (`OnnxExporter`)

The trained PyTorch model is exported to ONNX format with a fixed frequency sweep as part of the model signature. The resulting `.onnx` file is self-contained and can be run with `onnxruntime` — no PyTorch installation required at inference time.

### Stage 6 — Model testing (`ModelTester`)

The ONNX model is loaded and evaluated against held-out simulation data. Prediction errors are logged to help assess whether the surrogate is accurate enough for use in COBRA.

## Custom Geometry

To train a surrogate for your own component, create three files:

1. **A Python class** extending `BaseGeometry` — defines geometry parameters, GDS generation, and model architecture.
2. **A stackup XML file** — defines the physical layer stack (materials, thicknesses, conductor layers).
3. **A simulation config file** (`.simcfg`) — defines port positions, frequency sweep, and mesh settings for Palace.

See the [Custom Classes documentation](docs/custom_class.md) for a full walkthrough and examples.

The built-in `TransformerOcta` preset (`src/orca/geometry/presets/tf_octa_c_ports.py`) is a good reference implementation.

## Sharing Models on Hugging Face

After training a surrogate model with ORCA, you can publish it to [Hugging Face](https://huggingface.co) so that COBRA — or anyone else — can discover and use it directly.

### Requirements

- A Hugging Face account
- The `huggingface_hub` Python package: `pip install huggingface_hub`

### File structure

Each model repository must contain exactly two files named after the model:

| File | Description |
|------|-------------|
| `<model_name>.onnx` | The exported ONNX surrogate model produced by `OnnxExporter` |
| `<model_name>.py` | The Python geometry class (subclass of `BaseGeometry`) used to generate and train the model |

The geometry class file is required so that COBRA can reconstruct the parameter space, call back into the geometry for EM verification, and correctly pre-process inference inputs.

### Step-by-step upload

1. **Create a new model repository** at [https://huggingface.co/new](https://huggingface.co/new).  
   Set visibility to **Public** and note the repository ID (e.g. `your-username/tf-octa-c-ports`). Click on "Create model".

2. Create a **Model Card** (essentially just a structured README) for your repository. Click on "Add Model Card". From there, add the tag "orca-surrogate" to make it discoverable by COBRA and other users looking for ORCA. The model card should then include this section:

   ```markdown
   ---
   tags:
   - orca-surrogate
   ```

3. **Upload the files** using the `huggingface_hub` library:
   ```python
   from huggingface_hub import HfApi

   api = HfApi()
   repo_id = "your-username/tf-octa-c-ports"  # replace with your repo

   api.upload_file(path_or_fileobj="tf_octa_c_ports.onnx", path_in_repo="tf_octa_c_ports.onnx", repo_id=repo_id)
   api.upload_file(path_or_fileobj="tf_octa_c_ports.py",   path_in_repo="tf_octa_c_ports.py",   repo_id=repo_id)
   ```
   Or via the Hugging Face web interface: go to your repository → **Files** → **Add file → Upload files**.

4. **Verify** the repository contains both `<model_name>.onnx` and `<model_name>.py` and is tagged `orca-surrogate`.

### Using a shared model in COBRA

Once uploaded, COBRA can query all public `orca-surrogate` models or load a specific one directly by its Hugging Face repository ID. Refer to the [COBRA documentation](https://github.com/DI-PASSIONATE/COBRA) for details on how to point COBRA at a Hugging Face model repository.


## Troubleshooting

- If the `orca` command is not found, ensure your virtual environment is activated and reinstall with `pip install -e .`.
- If Palace simulations fail, verify Palace is installed and available in your `PATH`, or adjust the `palace_executable` argument.
- If GDS conversion fails, verify that [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2) is installed and that the stackup XML matches your technology.
- If `orca.ModelTrainer`, `orca.OnnxExporter` or `orca.ModelTester` raise `ModuleNotFoundError` (torch, sklearn, optuna, onnx...), the training dependencies are not installed: `pip install -e ".[train]"`.

## Cite This Work

If you use ORCA in your research, please cite our upcoming SBCCI 2026 paper (or use GitHub's **Cite this repository** button, backed by [`CITATION.cff`](CITATION.cff)):

```bibtex
@INPROCEEDINGS{2026_COBRA,
  author={Simone, Gianluca and Lurz, David and Grund, Martin and Schneider, Fabian and Loose, Michael and Breun, Sascha and Koch, Manuel and Weigel, Robert and Franchi, Norman},
  booktitle={2026 39th SBC/SBMicro/IEEE Symposium on Integrated Circuits and Systems Design (SBCCI)},
  title={{COBRA: An AI-Assisted Circuit-Level Optimizer for Open Source Based RFIC Design}},
  year={2026},
  organization={IEEE},
  keywords={artificial intelligence, design automation, EDA, neural network, open-source, optimization, Palace, radio frequency integrated circuit, surrogate model}
}
```

## Acknowledgements
This work was supported by the Bundesministerium für Forschung, Technologie und Raumfahrt (BMFTR) under the DI-PASSIONATE project. We thank our colleagues in the LITES institute for their feedback and support during development. Special thanks to the open-source community for providing the tools and libraries that made this project possible, including but not limited to:

- [gdsfactory](https://github.com/gdsfactory/gdsfactory)
- [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2)
- [setupEM](https://github.com/VolkerMuehlhaus/setupEM)
- [Palace](https://github.com/awslabs/palace)
- [PyTorch](https://github.com/pytorch/pytorch)
- [ONNX](https://github.com/onnx/onnx)
- [scikit-rf](https://github.com/scikit-rf/scikit-rf)
- [OpenStack](https://opendev.org/openstack)
- [Hugging Face Hub](https://github.com/huggingface/huggingface_hub)

<table width="100%">
  <tr>
    <td align="left" width="50%">
      <a href="https://www.lites.tf.fau.de/" target="_blank">
        <img src="docs/lites.png" alt="Lehrstuhl für Intelligente Technische Elektronik und Systeme @ FAU" width="94%"/>
      </a>
    </td>
    <td align="right" width="50%">
      <a href="https://www.elektronikforschung.de/projekte/di-passionate" target="_blank">
        <img src="docs/bmftr.jpg" alt="DI-PASSIONATE Project (funded by BMFTR)" width="94%"/>
      </a>
    </td>
  </tr>
</table>
