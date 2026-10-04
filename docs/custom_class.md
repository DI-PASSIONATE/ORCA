---
title: Custom Geometry Classes
description: >-
  Bring your own RF passive to ORCA: write a BaseGeometry subclass, a layer
  stackup XML and a Palace simulation config to generate GDS layouts, simulate
  them and train an S-parameter surrogate model.
---

# Custom Classes

If you want to simulate and predict your own geometry, you need to create three files:

- A python file extending the `BaseGeometry` class and implementing the required methods.
- A stackup XML file defining the layers of your geometry.
- A simulation configuration file defining the simulation parameters for Palace.

Once you have these three files, you can head over to [Running ORCA](running_orca.md) to see how to run ORCA with these files.

### Python Class
The Python class should be a `@dataclass` extending `orca.BaseGeometry` and must implement the following:

**Required dataclass fields:**

- `name: str` — Unique identifier for the geometry (used as directory name and file prefix).
- `stackup_xml: str` — Path to the stackup XML file describing the physical layer stack.
- `simconfig_filename: str` — Path to the Palace simulation configuration file (`.simcfg`).
- `input_parameter_iterator: InputParameterIterator` — Declares the geometry's parameters (`RangeParameter` / `ChoiceParameter` objects) and how they are sampled; see the [reference below](#reference-inputparameteriterator).

!!! warning "Give the object-valued field a `default_factory`"

    `input_parameter_iterator` must be declared with `field(default_factory=...)`,
    as in the example below. Writing `= InputParameterIterator(...)` instead builds
    **one** object shared by every instance of the class.

!!! tip "Choosing an output representation"

    The `codec` passed to the dataset decides what the model regresses. `FlatReImCodec`
    predicts all `N x N` entries; `UpperTriangleReImCodec` predicts only the upper
    triangle and mirrors it on decode, which halves the output dimension and makes
    `S = S^T` structural. Use it for reciprocal passives (any ordinary inductor or
    transformer), and `FlatReImCodec` for anything non-reciprocal. Note that this
    changes the ONNX output names. See `src/orca/training/README.md`.

!!! note "Basis expansions are not set on the geometry"

    Engineered inputs such as a Chebyshev expansion of frequency belong to the
    model, not the geometry: pass `orca.ModelTrainer(model=..., basis="chebyshev")`.
    That way the expansion is tuned with the model and traced into the exported ONNX
    graph. See `src/orca/training/README.md`.

**Required abstract methods:**

- `create_gds_file(name, output_path, params) -> str` — Generates a GDS layout file from geometry parameters. Returns the path to the created file.
- `create_dataset() -> BaseDataset` — Builds the dataset (e.g. `GeoToSParamDatasetSingleFrequency`) with its output codec and normalizers, used for training. It is called once per geometry instance, the first time `geometry.dataset` is read, so each instance gets its own dataset and normalizer statistics. A `MinMaxNormalizer` expects the parameter table's columns in the order of the input parameter iterator, with `frequency` last; `ModelTrainer` reorders the table to match and rejects one with missing or extra columns. A custom `BaseDataset` subclass appends the result file name of every sample it loads to `self.sample_groups`, so cross-validation can keep each geometry on one side of a fold.

**Optional methods:**

- `electrical_parameters(ntwk) -> dict[str, np.ndarray]` — The figures of merit of a network of this geometry, which `ModelTester` reports the model's error for next to the S-parameter error: curves over `ntwk.f` (such as L, R and Q) and scalars as 0-d arrays (such as the self-resonance frequency `srf_f`). Which port is which is a property of the layout, so the geometry decides how to read its network. `orca.utils.postprocessing` has the usual ones, with the ports given explicitly: `inductor_parameters(ntwk, ends, shorted)` and `transformer_parameters(ntwk, primary, secondary, shorted)`, which drive each winding differentially with its center tap (`shorted`) AC-grounded. Import it inside the method. The default returns nothing, so only the S-parameter error is reported. Errors are relative, except for the names in the class attribute `absolute_error_parameters` (e.g. `frozenset({"k"})` for a coupling factor that is near zero for weak coupling). With `ModelTester(plot=True)` the same parameters are plotted, reference against prediction.
- `feasibility_constraints() -> list[str]` — The same rules as `is_feasible()`, written as boolean expressions over the input parameter names, e.g. `"bottom_linewidth <= bottom_winding_diameter / 3"`. `OnnxExporter` stores them in the model's `input_constraints` metadata, so COBRA can refuse a query for a geometry that cannot be built instead of returning a prediction the model was never trained for. The grammar is a small subset of Python (arithmetic, comparisons, `and`/`or`/`not`, `a if c else b`, and `abs min max sqrt sin cos tan radians ceil floor round`, plus `pi` and `sqrt2`), documented in `orca.geometry.constraints`; anything else is rejected at export. Derive the strings from the same numbers as `is_feasible()` and add a test that they agree on random draws (see `tests/test_constraints.py` for the presets' version).
- `is_feasible(params) -> bool` — Whether a parameter combination describes a layout that can be drawn (default: always `True`). `GDSGenerator` calls it for every draw of the iterator; rejected draws are counted and, with the `"sobol"`, `"lhs"` and `"random"` strategies, replaced by further draws, so the requested number of samples is met with buildable layouts only. Put cheap, closed-form constraints between parameters here — a winding that must fit its diameter, a feed gap that must fit the octagon's side. The presets derive it from the same check their cell code runs, so the two cannot disagree.

!!! warning "Reject, never clamp"

    Do not repair a bad parameter inside `create_gds_file()` (for example clamp a
    too-small diameter to the buildable minimum). The parameter table records the
    *requested* values, so a repaired layout trains the model on a geometry it does
    not have, and the surrogate then returns confident results for inputs that
    were never built. Reject the draw in `is_feasible()` and raise `ValueError`
    in `create_gds_file()` as the safety net.

!!! note "Grid snapping and design rules are not the geometry's job"

    Draw the layout and return; do not snap vertices to the manufacturing grid or
    re-implement design-rule checks in `create_gds_file()`. The `DRCChecker` stage
    snaps every generated GDS file to the SG13G2 grid and drops layouts that break
    the PDK's metal and via rules (see [Pipeline Stages](pipeline.md)). It is the
    safety net behind `is_feasible()`, not a substitute for it: a rule that can be
    written down belongs in `is_feasible()`, where it costs nothing and keeps the
    sample count honest.

!!! tip "Keep the training imports inside `create_dataset()`"

    The dataset classes and normalizers need PyTorch, which is an optional
    dependency of ORCA (the `train` extra). Import them inside `create_dataset()`,
    as in the example below, and your geometry can still generate layouts and run
    Palace simulations in an environment without PyTorch (e.g. an HPC cluster).

The model architecture and its hyperparameters are not part of the geometry: pass them to
the training stage with `orca.ModelTrainer(model=..., hyperparameters=...)`, or leave
`hyperparameters` out to let `ModelTrainer` tune them with Optuna over the model's own search space.

Example:

```python
# Builds a fresh iterator per geometry instance. See the warning above:
# a bare `= InputParameterIterator(...)` default would be shared by every instance.
def _input_parameters() -> InputParameterIterator:
    # All in µm, on a 0.1 µm grid (from orca import InputParameterIterator, RangeParameter)
    return InputParameterIterator(
        RangeParameter("bottom_winding_diameter", 20.0, 120.0, step=0.1),
        RangeParameter("top_winding_diameter", 20.0, 120.0, step=0.1),
        # The offset between the winding centres as a share of their mean diameter
        RangeParameter("relative_displacement", 0.0, 0.2, step=0.005),
        RangeParameter("bottom_linewidth", 2.0, 12.0, step=0.1),
        RangeParameter("top_linewidth", 2.0, 12.0, step=0.1),
        picking_strategy="sobol",
        frequency=[1e9, 500e9],  # 1 GHz to 500 GHz
    )


@dataclass
class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape.
    """

    name: str = "tf_octa_c_ports"
    stackup_xml: str = StackupXML.SG13G2_FEM_200um  # from orca.geometry.presets
    simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")
    input_parameter_iterator: InputParameterIterator = field(default_factory=_input_parameters)

    def create_dataset(self) -> "BaseDataset":
        # Imported here so the geometry works without PyTorch (see the tip above).
        from orca.training.codecs import FlatReImCodec
        from orca.training.datasets.geo_to_s_param_single_f import GeoToSParamDatasetSingleFrequency
        from orca.training.normalize import MinMaxNormalizer, StandardNormalizer

        return GeoToSParamDatasetSingleFrequency(
            codec=FlatReImCodec(n_ports=6),
            # Scale inputs with the declared parameter ranges rather than the
            # min/max of whatever subset happens to be trained on.
            input_normalizer=MinMaxNormalizer(self.input_parameter_iterator),
            output_normalizer=StandardNormalizer(),
        )

    # Close to zero for weakly coupled windings, where a relative error says nothing
    absolute_error_parameters: ClassVar[frozenset[str]] = frozenset({"k"})

    def electrical_parameters(self, ntwk: "rf.Network") -> dict[str, "np.ndarray"]:
        from orca.utils.postprocessing import transformer_parameters

        # Zero-based ports: 0 op, 1 on (top winding), 2 ip, 3 in (bottom winding),
        # 4 and 5 the center taps, AC-grounded as in differential use
        return transformer_parameters(ntwk, primary=(0, 1), secondary=(2, 3), shorted=(4, 5))

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:
        # 
        # < Put your actual GDS generation code here >
        #
        c.write_gds(output_path, with_metadata=False)
        return output_path
```

### Reference: InputParameterIterator

`InputParameterIterator` declares the geometry's parameters and how they are sampled during GDS generation. Each parameter is an object; their order is the column order of the parameter table and of the model inputs.

```python
from orca import ChoiceParameter, InputParameterIterator, RangeParameter

InputParameterIterator(
    RangeParameter("turns", 1, 5, dtype=int),                     # 1, 2, 3, 4, 5
    RangeParameter("width", 2.0, 15.0, step=0.02, sampling="log"),
    RangeParameter("diameter", 30.0, 300.0, step=2.0, sampling="log"),
    RangeParameter("angle", 0.0, 45.0),                           # continuous
    ChoiceParameter("layer", [126, 134]),
    picking_strategy="sobol",   # "sobol" (default), "lhs", "random", "grid" / "uniform_grid", "step_grid"
    frequency=[1e9, 500e9],     # Optional: frequency band in Hz, for normalisation (not drawn)
    boundary_fraction=0.05,     # Share of draws moved onto the faces, edges and corners of the box
    kept_on_rejection=("diameter",),  # Kept when is_feasible rejects a draw; the rest is redrawn
)
```

`RangeParameter(name, min, max, step=None, sampling="uniform", dtype=float)`:

| Argument | Meaning |
|---|---|
| `min`, `max` | Bounds, both inclusive. `min == max` makes a fixed parameter. |
| `step` | Spacing of the allowed values `min`, `min + step`, … `max`; it must divide `max - min`. Values are rounded to the step's decimals (a 0.1 step gives 0.3, not 0.30000000000000004). Without a step the parameter is continuous. |
| `sampling` | `"uniform"` spreads draws evenly over the range. `"log"` spreads them evenly in the logarithm — as many draws between 30 and 100 as between 100 and 300 — which favours small values; it needs `min > 0`. |
| `dtype` | `float` or `int`. An integer parameter steps by 1 unless given another integer step, and `create_gds_file` receives Python `int`s. |

`ChoiceParameter(name, values)` takes one of an explicit list of values, each equally likely.

Picking strategies:

| Strategy | Behaviour |
|---|---|
| `"sobol"` (default) | A scrambled Sobol' sequence: a space-filling design that spreads the samples over the parameter box much more evenly than random draws, so the same number of simulations leaves fewer gaps. Reproducible with the run seed. |
| `"lhs"` | Latin hypercube designs of `num_samples` points: each parameter's range is cut into `num_samples` strata with one sample each. Evener than random, less so than Sobol' in several dimensions together. |
| `"random"` | Draws every parameter independently from its sampling distribution (with replacement). |
| `"grid"` / `"uniform_grid"` | About `num_samples ** (1 / number of parameters)` evenly spaced quantiles of each parameter's distribution (always including its min and max), combined. A log-sampled parameter is gridded on a log scale. |
| `"step_grid"` | Every value of every parameter combined; all parameters need a `step` or be choices. |

With the three drawing strategies, draws `is_feasible()` rejects are replaced by further draws, and `boundary_fraction` (default 0.05) of the draws are moved onto the boundary of the box: a random number of their parameters, at least one, is set to its min or max. Space-filling and random designs almost never reach the faces, edges and corners themselves, yet a network extrapolates worst there and an optimizer often ends up there. Set it to 0 to switch this off. The grid strategies include each parameter's min and max anyway.

A rejection skews the accepted samples towards the values that are feasible most often: an inductor's small diameters only fit one turn and thin lines, so most small-diameter draws are rejected, and the accepted diameters pile up at the large end. Parameters listed in `kept_on_rejection` keep their drawn value instead, and the other parameters are redrawn around it (up to 100 times) until the combination is feasible, so those parameters follow the distribution they are declared with. The `InductorOcta` preset keeps its diameter this way.

`GDSGenerator` plots the result to `geometries/<name>_coverage.png`: every pair of parameters with the layouts over the rejected draws, so gaps in the sampling can be told apart from regions `is_feasible()` excludes.

The ranges, not the sampling, define what the model covers: `input_parameter_ranges` in the exported ONNX model holds each parameter's `min` and `max`. Sampling only decides where in that box the simulations are concentrated.

### Reference: Dataset Types

| Class | Description |
|---|---|
| `GeoToSParamDatasetSingleFrequency` | One training sample per frequency point per geometry (recommended) |
| `GeoToSParamDataset` | One training sample per geometry (full frequency sweep as a vector) |

### Reference: Basis Expansions

Basis expansions widen the model's input before its first layer by appending fixed
basis functions of it. Nothing about them is learned. They belong to the model,
not the geometry, so they are chosen on the training stage — `orca.ModelTrainer(model="mlp",
basis="chebyshev")` — and work with any architecture. Because the expansion is part
of the model, it is tuned with it and traced into the exported ONNX graph; the ONNX
input names stay the same.

| Class | Name | Description |
|---|---|---|
| `IdentityBasis` | `"identity"` | Passes inputs through unchanged (the default) |
| `ChebyshevBasis` | `"chebyshev"` | Appends `degree` Chebyshev polynomials of one column, `frequency` by default |

`ChebyshevBasis` tunes `basis_degree` automatically when no hyperparameters are
supplied. It sees *normalized* inputs and clamps to [-1, 1], so a frequency outside
the training range saturates rather than diverging.

### Reference: Normalizers

**Input normalizers** (passed as `input_normalizer` to the dataset):

| Class | Description |
|---|---|
| `MinMaxNormalizer(input_parameter_iterator)` | Min-max normalisation with the declared parameter ranges (recommended: independent of which samples are trained on, and matches the `input_parameter_ranges` metadata of the exported ONNX model) |
| `OutputMinMaxNormalizer` | Min-max normalisation fitted to the min/max of the training split |

**Output normalizers** (passed as `output_normalizer` to the dataset):

| Class | Description |
|---|---|
| `StandardNormalizer` | Z-score normalisation (zero mean, unit variance) |
| `OutputMinMaxNormalizer` | Min-max normalisation |

### Stackup XML File
The stackup XML describes the physical layer stack for [gds2palace](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2): materials, dielectric thicknesses, and which GDS layer is drawn as which conductor, via or sheet.

The presets use unmodified copies of gds2palace's current IHP stackups, in `src/orca/geometry/presets/stackups/`. Reference them through the `StackupXML` enum, whose members are the absolute paths as `str`:

```python
from orca.geometry.presets import StackupXML

stackup_xml: str = StackupXML.SG13G2_FEM_200um
```

| `StackupXML` member | File | Technology |
|---|---|---|
| `SG13G2_FEM_200um` | `SG13G2_FEM_200um.xml` | SG13G2, planar SiO2 + passivation over TopMetal2, 200 µm chip |
| `SG13G2_FEM_200um_passi3D` | `SG13G2_FEM_200um_passi3D.xml` | SG13G2, conformal SiO2 around TopMetal2 (derived layers), 200 µm chip |

The gds2palace package on PyPI does not ship stackup files. For another technology or variant, copy one from the [`XML_stackup/latest`](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/tree/main/XML_stackup/latest) folder of the gds2palace repository, next to your geometry class, and point `stackup_xml` at it. Avoid the files in `XML_stackup/legacy`; they are kept upstream only for older models. The format is described in the [XML stackup format description](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/blob/main/doc/XML_stackup_format/XML_stackup_format.md).

### Simulation Configuration File
The simulation configuration file (.simcfg) defines how to mesh and run the electromagnetic simulation in Palace. 
You can either tweak the preset .simcfg files next to their geometry classes in `src/orca/geometry/presets/inductor/` and `src/orca/geometry/presets/transformer/`, create your own from scratch,
or use the GUI provided by [setupEM](https://github.com/VolkerMuehlhaus/setupEM/tree/main) to create the .simcfg file interactively.

Example:

```json
{
    "application": "setupEM",
    "data_format": "1.0",
    "saved_values": {
        "preprocess_gds": true,
        "merge_polygon_size": 0.5,
        "purpose": [
            0
        ],
        "fstart": 0.0,
        "fstop": 170.0,
        "refined_cellsize": 2.0,
        "order": 2,
        "cells_per_wavelength": 20.0,
        "meshsize_max": 100.0,
        "adaptive_mesh_iterations": 0,
        "iterative": false,
        "boundary": [
            "PEC",
            "PEC",
            "PEC",
            "PEC",
            "PEC",
            "PEC"
        ],
        "margin": 200.0,
        "air_around": 200.0,
        "ELMER_MPI_THREADS": 4,
        "model_basename": "tf_octa_c_ports",
        "sim_path": "/home/users/simone/OpenSource_LNA",
        "fstep": 1.0
    },
    "ports": [
        {
            "portnumber": 1,
            "source_layernum": 201,
            "target_layername": null,
            "from_layername": "Metal5",
            "to_layername": "TopMetal1",
            "direction": "Z",
            "port_Z0": 50.0,
            "voltage": 1.0
        },
        {
            "portnumber": 2,
            "source_layernum": 204,
            "target_layername": null,
            "from_layername": "Metal5",
            "to_layername": "TopMetal1",
            "direction": "Z",
            "port_Z0": 50.0,
            "voltage": 1.0
        },
        {
            "portnumber": 3,
            "source_layernum": 202,
            "target_layername": null,
            "from_layername": "Metal5",
            "to_layername": "TopMetal2",
            "direction": "Z",
            "port_Z0": 50.0,
            "voltage": 1.0
        },
        {
            "portnumber": 4,
            "source_layernum": 203,
            "target_layername": null,
            "from_layername": "Metal5",
            "to_layername": "TopMetal2",
            "direction": "Z",
            "port_Z0": 50.0,
            "voltage": 1.0
        }
    ]
}
```
