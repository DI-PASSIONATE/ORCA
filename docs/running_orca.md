# Quickstart

Once you have completed installation (see **Getting Started -> Installation**), you are ready to run ORCA.

## 1. Prepare Inputs

You need:

- A geometry class (`BaseGeometry` subclass) with parameter definitions.
- A stackup XML file describing the physical layer stack.
- A Palace simulation configuration file (`.simcfg`).
- A Palace executable (local or via Apptainer container).

For an immediate starting point, use the built-in `TransformerOcta` geometry from `examples/`.

## 2. Run GUI Mode

```bash
orca
```

In GUI mode:

1. Select a geometry class.
2. Configure pipeline stages and parameters.
3. Set Palace executable path.
4. Start the pipeline.

## 3. Run Script Mode

For direct integration into scripts or custom workflows:

```python
import orca
from orca import ORCA
from orca.geometry.examples.transformer.tf_octa_c_ports import TransformerOcta

geometry = TransformerOcta()

orca_instance = ORCA(
    [
        orca.GDSGenerator(num_samples=1000),
        orca.GDSConverter(),
        orca.PalaceSimulator(
            palace_executable="apptainer exec ~/palace/palace.sif palace",
            touchstone_type="dc_deembedded",  # "all", "normal", "deembedded", "dc", "dc_deembedded"
        ),
        orca.ModelTrainer(
            # hyperparameters=None,   # If None, Optuna tunes automatically
            # test_frac=0.15,         # Fraction of data held out for testing
            # n_train_samples=None,   # Optional cap on training samples
            # n_fold_cv=5,            # Cross-validation folds during tuning
        ),
        orca.OnnxExporter(),
        orca.ModelTester(),
    ]
)

orca_instance.run(geometry=geometry, cpu_cores=16)
```

This will:

1. Generate 1000 parameterised GDS layout variants.
2. Convert each to a Palace mesh and run full-wave EM simulation.
3. Store results in Touchstone format under `output/<geometry_name>/`.
4. Train a neural network on the simulation dataset.
5. Export the trained model to ONNX format.
6. Test prediction accuracy against held-out simulation data.

!!! tip
	You can run only a subset of pipeline stages by modifying the list passed to `ORCA(...)`. Stages are sorted by their internal index and some depend on outputs of earlier stages. Stage indices: `GDSGenerator=0`, `GDSConverter=1`, `PalaceSimulator=2`, `ModelTrainer=4`, `OnnxExporter=5`, `ModelTester=6`.

## 4. Run at Scale with OpenStack

For large-scale simulation campaigns, we provide an OpenStack VM image and a REST API with corresponding client CLI.
See the [ORCA-OpenStack repository](https://github.com/DavidL-11/ORCA-OpenStack) for details.

## 5. Inspect Results

Each run stores artifacts under `output/<geometry_name>/`:

```text
output/<geometry_name>/
    context.json
    geometries/
    models/
    palace_sims/
    results/
```

Typical contents include:

- Touchstone `.sNp` files per simulated sample.
- PyTorch model checkpoint.
- Exported `.onnx` surrogate model.
- Full run context JSON.

!!! tip
	The exported `.onnx` file can be used directly with COBRA for circuit-level RFIC optimization.