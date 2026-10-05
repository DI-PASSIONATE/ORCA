# ORCA — Open RF Integrated Circuit Automation

**An open-source EDA tool for AI-assisted RFIC design: a surrogate modelling pipeline for RF integrated circuit components — from parametric GDS layout to full-wave EM simulation to a trained ONNX model for COBRA.**

[![Documentation](https://img.shields.io/badge/docs-di--passionate.github.io%2FORCA-green?logo=materialformkdocs&logoColor=white)](https://di-passionate.github.io/ORCA/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/DI-PASSIONATE/ORCA/blob/main/LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue?logo=python&logoColor=white)](https://github.com/DI-PASSIONATE/ORCA/blob/main/pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/orca-rfic?logo=pypi)](https://pypi.org/project/orca-rfic/)
[![Cite this repository](https://img.shields.io/badge/cite-CITATION.cff-lightgrey)](https://github.com/DI-PASSIONATE/ORCA/blob/main/CITATION.cff)
[![GitHub stars](https://img.shields.io/github/stars/DI-PASSIONATE/ORCA?style=social)](https://github.com/DI-PASSIONATE/ORCA/stargazers)

©2026

Gianluca Simone\*, David Lurz\*, Martin Grund\*, Fabian Schneider°, Michael Loose\*, Sascha Breun\*, Manuel Koch\*, Robert Weigel\*, Norman Franchi\*

\* Institute for Intelligent Electronics and Systems (LITES), Friedrich-Alexander-Universität (FAU), Erlangen-Nürnberg, Germany

° Chair of Integrated Electronic Systems, Otto-von-Guericke-University Magdeburg, Germany

[Paper (Coming Soon)](#cite-this-work) | [Documentation](https://DI-PASSIONATE.github.io/ORCA/) | [BibTeX](#cite-this-work)

> [!NOTE]
> ORCA is still under active development. The current codebase is functional and can be used for experimentation, but we keep adding features, improving documentation, and refining the API. If you encounter any issues or have questions, please [open an issue](https://github.com/DI-PASSIONATE/ORCA/issues) or reach out.

**ORCA** builds neural-network surrogate models of RF integrated circuit components, such as on-chip inductors and transformers. Instead of running a slow electromagnetic (EM) simulation for every candidate geometry during circuit design, ORCA runs the simulations once and learns how the geometry parameters map to S-parameters. It hands the result to circuit optimizers like [COBRA](https://github.com/DI-PASSIONATE/COBRA) as a portable ONNX model.

Given a parametric geometry class, ORCA:

1. generates thousands of GDS layout variants with [gdsfactory](https://github.com/gdsfactory/gdsfactory) and drops those that fail the design rules, checked with [KLayout](https://www.klayout.de/),
2. meshes them with [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2) and [gmsh](https://gmsh.info/), then runs full-wave EM simulations with [Palace](https://github.com/awslabs/palace),
3. trains a [PyTorch](https://pytorch.org/) model to predict S-parameters from the geometry and frequency, with hyperparameters tuned by [Optuna](https://optuna.org/),
4. exports it to [ONNX](https://onnx.ai/) and tests it against held-out simulations.

The whole flow is built on open-source tools, from layout to trained model.

![ORCA pipeline overview: parametric GDS generation, Palace EM simulation, PyTorch training and ONNX export of an RFIC component surrogate model](https://github.com/DI-PASSIONATE/ORCA/raw/main/docs/orca.png)

## Key Features

- **Parametric layout generation**: sample thousands of buildable GDS variants from a small Python geometry class.
- **Open-source full-wave EM simulation**: S-parameters from [Palace](https://github.com/awslabs/palace) on a laptop, a workstation or a Slurm HPC cluster.
- **Surrogate models**: PyTorch models with automatic normalization and Optuna hyperparameter tuning.
- **Portable ONNX export**: the model runs with `onnxruntime` alone; COBRA loads it directly.
- **Built-in validation**: error statistics of the exported model on held-out geometries.
- **GUI and scripting**: click through the pipeline or drive it from Python.
- **Model sharing**: publish surrogates on the Hugging Face Hub for others to reuse.
- **Technology-agnostic**: bring your own layer stackup XML.

ORCA builds the model; [COBRA](https://github.com/DI-PASSIONATE/COBRA) uses it to optimize circuits without running EM simulations, and can call back into ORCA's geometry classes to verify a design with EM.

## Installation

ORCA needs Python 3.11–3.13 and, for the EM simulations, [Palace](https://awslabs.github.io/palace/stable/install/index.html). It is published on PyPI as [`orca-rfic`](https://pypi.org/project/orca-rfic/); the import package and the GUI command are called `orca`.

**From PyPI**, with pip or uv in a virtual environment:

```bash
pip install "orca-rfic[train]"      # full pipeline
pip install orca-rfic               # layout generation and simulation only, no PyTorch

uv pip install "orca-rfic[train]"   # the same with uv
```

**From source** with [uv](https://docs.astral.sh/uv/) (recommended for development):

```bash
git clone https://github.com/DI-PASSIONATE/ORCA
cd ORCA
uv sync --extra train --extra cpu     # CPU-only PyTorch
uv sync --extra train --extra cu130   # or CUDA 13.0 (driver >= 580); cu126 for older drivers
uv sync                               # or simulation only, no PyTorch
```

Run ORCA with `uv run orca` or after `source .venv/bin/activate`. The [installation guide](https://di-passionate.github.io/ORCA/setup/) also covers installing from source with pip; [HPC_INSTALL.md](https://github.com/DI-PASSIONATE/ORCA/blob/main/examples/slurm_runs/HPC_INSTALL.md) shows an install on a Slurm cluster.

## Quickstart

Start the GUI with:

```bash
orca
```

Or run the pipeline from a Python script:

```python
import orca
from orca.geometry.presets import TransformerOcta

pipeline = orca.ORCA(
    [
        orca.GDSGenerator(num_samples=1000),
        orca.DRCChecker(),
        orca.GDSConverter(),
        orca.PalaceSimulator(palace_executable="palace"),
        orca.ModelTrainer(),
        orca.OnnxExporter(),
        orca.ModelTester(),
    ]
)

if __name__ == "__main__":  # required: worker processes re-import this script
    pipeline.run(geometry=TransformerOcta(), num_processes=16, seed=40)
```

Each stage can be left out, e.g. to retrain on existing simulation results. An interrupted run picks up where it stopped when started again. See the [quickstart](https://di-passionate.github.io/ORCA/running_orca/) for every stage option.

## Pipeline Stages

| Stage | Class | What it does |
|-------|-------|--------------|
| GDS generation | `GDSGenerator` | Samples the geometry's parameters and draws a GDS layout for each |
| Design-rule check | `DRCChecker` | Snaps layouts to the manufacturing grid and drops those that violate the IHP SG13G2 rules or have a port marker off its metal |
| GDS conversion | `GDSConverter` | Meshes the layouts for Palace with [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2) |
| EM simulation | `PalaceSimulator` | Runs Palace and stores the S-parameters as Touchstone files |
| Model training | `ModelTrainer` | Trains (and by default tunes) a PyTorch model from geometry and frequency to S-parameters |
| ONNX export | `OnnxExporter` | Exports the model to a self-contained ONNX file for COBRA |
| Model testing | `ModelTester` | Reports the model's error on held-out geometries |

The [pipeline documentation](https://di-passionate.github.io/ORCA/pipeline/) describes each stage in detail.

## Custom Geometries

To build a surrogate of your own component, write a `BaseGeometry` subclass that defines its parameters and draws its layout, and pair it with a layer stackup XML and a Palace simulation config (`.simcfg`). The [custom classes guide](https://di-passionate.github.io/ORCA/custom_class/) walks through it; the bundled presets (`InductorOcta`, `TransformerOcta`) are reference implementations.

## Sharing Models

Trained models can be published on the Hugging Face Hub with the `orca-rfic` tag, where COBRA and other users can find them. See [Sharing Models on Hugging Face](https://di-passionate.github.io/ORCA/sharing_models/).

## Documentation

The full documentation is at [di-passionate.github.io/ORCA](https://di-passionate.github.io/ORCA/), including [troubleshooting](https://di-passionate.github.io/ORCA/troubleshooting/).

## Cite This Work

If you use ORCA in your research, please cite our upcoming SBCCI 2026 paper (or use GitHub's **Cite this repository** button, backed by [`CITATION.cff`](https://github.com/DI-PASSIONATE/ORCA/blob/main/CITATION.cff)):

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
- [Hugging Face Hub](https://github.com/huggingface/huggingface_hub)

The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU). The hardware is partially funded by the German Research Foundation (DFG).

<table width="100%">
  <tr>
    <td align="left" width="50%">
      <a href="https://www.lites.tf.fau.de/" target="_blank">
        <img src="https://github.com/DI-PASSIONATE/ORCA/raw/main/docs/lites.png" alt="Lehrstuhl für Intelligente Technische Elektronik und Systeme @ FAU" width="94%"/>
      </a>
    </td>
    <td align="right" width="50%">
      <a href="https://www.elektronikforschung.de/projekte/di-passionate" target="_blank">
        <img src="https://github.com/DI-PASSIONATE/ORCA/raw/main/docs/bmftr.jpg" alt="DI-PASSIONATE Project (funded by BMFTR)" width="94%"/>
      </a>
    </td>
  </tr>
</table>
