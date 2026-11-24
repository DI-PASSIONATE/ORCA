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

    `FlatReImCodec` predicts all `N x N` S-parameters. `UpperTriangleReImCodec` predicts the
    upper triangle and mirrors it, for reciprocal passives: half the outputs, and `S = S^T`
    by construction. It changes the ONNX output names. See `src/orca/training/README.md`.

!!! note "Basis expansions are not set on the geometry"

    Pass them to the training stage: `orca.ModelTrainer(model=..., basis="chebyshev")`.

**Required abstract methods:**

- `create_gds_file(name, output_path, params) -> str` — Generates a GDS layout file from geometry parameters. Returns the path to the created file.
- `create_dataset() -> BaseDataset` — Builds the dataset (e.g. `GeoToSParamDatasetSingleFrequency`) with its output codec and normalizers. Called once per geometry instance. The parameter table must have exactly the iterator's columns plus `frequency`. A custom `BaseDataset` appends the result file name of every sample it loads to `self.sample_groups`, so cross-validation can split by geometry.

**Optional methods:**

- `is_feasible(params) -> bool` — Whether a layout with these parameters can be drawn (default: always). `GDSGenerator` replaces rejected draws. Put cheap, closed-form constraints between parameters here, such as a winding that must fit its diameter.
- `feasibility_constraints() -> list[str]` — The same rules as expressions over the parameter names, e.g. `"bottom_linewidth <= bottom_winding_diameter / 3"`, for the ONNX metadata, so COBRA rejects geometries that cannot be built. The grammar is documented in `orca.geometry.constraints`. Add a test that they agree with `is_feasible()` (see `tests/test_constraints.py`).
- `simulation_ports(params) -> list[dict]` — The ports of one sample (default: the simconfig's). Override it when the metal a port lands on depends on the parameters, as in `InductorOcta`. Change port fields only, never a port's number or order.
- `windings() -> list[list[int]]` — The ports (1-based) of each winding, e.g. `[[1, 2, 5], [3, 4, 6]]` for `TransformerOcta`. COBRA then replaces the predicted 0 Hz point by that of ideal windings. Leave it out for a component with an intended DC path to ground or between windings.
- `electrical_parameters(ntwk) -> dict[str, np.ndarray]` — Figures of merit such as L, Q and SRF, whose errors `ModelTester` reports: curves over `ntwk.f`, or scalars as 0-d arrays. `orca.utils.postprocessing` has `inductor_parameters(ntwk, ends, shorted)` and `transformer_parameters(ntwk, primary, secondary, shorted)`; ports in `shorted` (e.g. center taps) are AC-grounded, those in `terminations={port: impedance}` loaded to ground, and others open. Import them inside the method. List parameters that can be near zero in `absolute_error_parameters`, e.g. `frozenset({"k"})`.

!!! warning "Reject, never clamp"

    Do not repair a bad parameter inside `create_gds_file()`. The parameter table records
    the *requested* values, so the model would learn a geometry that was not built.
    Reject the draw in `is_feasible()`, and raise `ValueError` in `create_gds_file()` as
    the safety net.

!!! note "Grid snapping and design rules are not the geometry's job"

    Do not snap vertices or re-implement design-rule checks in `create_gds_file()`:
    `DRCChecker` does both (see [Pipeline Stages](pipeline.md)). A rule you can write down
    belongs in `is_feasible()`, where it costs nothing.

!!! tip "Keep the training imports inside `create_dataset()`"

    Training needs PyTorch, which is optional. With its imports inside `create_dataset()`,
    the geometry still generates layouts and runs Palace without it, e.g. on an HPC cluster.

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
    stackup_xml: str = StackupXML.SG13G2_FEM_200um_passi3D  # from orca.geometry.presets
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
| `"sobol"` (default) | Scrambled Sobol' sequence: covers the parameter box more evenly than random draws. |
| `"lhs"` | Latin hypercube: one sample in each of `num_samples` strata of every parameter. |
| `"random"` | Independent draws from each parameter's distribution. |
| `"grid"` / `"uniform_grid"` | About `num_samples ** (1 / number of parameters)` quantiles of each parameter, combined. |
| `"step_grid"` | Every value of every parameter combined; all need a `step` or be choices. |

The first three replace draws that `is_feasible()` rejects. Two options adjust them:

- `boundary_fraction` (default 0.05) moves that share of the draws onto the faces, edges and corners of the box, which such designs almost never reach. 0 switches it off.
- `kept_on_rejection` lists parameters that keep their value while the others are redrawn. Without it, values that are rarely feasible, such as small diameters, end up under-represented.

`input_parameter_ranges` in the exported model holds each parameter's `min` and `max` (and the `step` of integers); sampling only decides where in that box the simulations are.

### Reference: Dataset Types

| Class | Description |
|---|---|
| `GeoToSParamDatasetSingleFrequency` | One training sample per frequency point per geometry (recommended) |
| `GeoToSParamDataset` | One training sample per geometry (full frequency sweep as a vector) |

### Reference: Basis Expansions

Basis expansions append fixed functions of the inputs before the model's first layer. They are part of the model, so they are chosen on the training stage (`orca.ModelTrainer(model="mlp", basis="chebyshev")`), tuned with it and exported in the ONNX graph.

| Class | Name | Description |
|---|---|---|
| `IdentityBasis` | `"identity"` | Passes inputs through unchanged (the default) |
| `ChebyshevBasis` | `"chebyshev"` | Appends `degree` Chebyshev polynomials of one column, `frequency` by default |

`basis_degree` is tuned unless hyperparameters are given. Inputs outside the training range saturate rather than diverge.

### Reference: Normalizers

**Input normalizers** (passed as `input_normalizer` to the dataset):

| Class | Description |
|---|---|
| `MinMaxNormalizer(input_parameter_iterator)` | Min-max normalisation with the declared parameter ranges (recommended) |
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

The copies keep a gds2palace update from changing a dataset's stackup. For another technology, use a file gds2palace ships (`gds2palace.utilities.get_stackup_path("SG13CMOS5L_200um.xml")`, since 0.9.1) or copy one from its [`XML_stackup/latest`](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/tree/main/XML_stackup/latest) folder next to your geometry class. See the [format description](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/blob/main/doc/XML_stackup_format/XML_stackup_format.md).

### Simulation Configuration File
The simulation configuration file (.simcfg) defines how to mesh and run the electromagnetic simulation in Palace. 
You can either tweak the preset .simcfg files next to their geometry classes in `src/orca/geometry/presets/inductor/` and `src/orca/geometry/presets/transformer/`, create your own from scratch,
or use the GUI provided by [setupEM](https://github.com/VolkerMuehlhaus/setupEM/tree/main) to create the .simcfg file interactively.

Frequencies are in GHz: a linear sweep from `fstart` to `fstop` in steps of `fstep`, plus the points in `fpoint`, which are nearly free with Palace's adaptive sweep. Palace cannot solve at DC: with `"fstart": 0.0` it solves 10 and 20 MHz instead, and the DC point of the `dc` Touchstone files is extrapolated from there. Set the geometry's `frequency=` band to what the simcfg sweeps, starting at 0 if the results include the DC point.

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

The `ports` list applies to every sample, unless the geometry overrides `simulation_ports(params)`. Each port's marker on its `source_layernum` must touch the metals it connects; `DRCChecker` drops layouts where it does not.
