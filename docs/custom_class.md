# Custom Classes

If you want to simulate and predict your own geometry, you need to create three files:

- A python file extending the `BaseGeometry` class and implementing the required methods.
- A stackup XML file defining the layers of your geometry.
- A simulation configuration file defining the simulation parameters for Palace.

Once you have these three files, you can head over to [Running ORCA](/docs/running_orca.md) to see how to run ORCA with these files.

### Python Class
The Python class should be a `@dataclass` extending `orca.BaseGeometry` and must implement the following:

**Required dataclass fields:**

- `name: str` — Unique identifier for the geometry (used as directory name and file prefix).
- `stackup_xml: str` — Path to the stackup XML file describing the physical layer stack.
- `simconfig_filename: str` — Path to the Palace simulation configuration file (`.simcfg`).
- `input_parameter_iterator: InputParameterIterator` — Defines geometry parameters and their sampling ranges.
- `dataset: BaseDataset` — Dataset class instance (e.g. `GeoToSParamDatasetSingleFrequency`) used for training.

**Optional dataclass fields:**

- `features: FeatureTransformPipeline | None` — Optional pipeline of engineered features (ratios, Chebyshev polynomials, etc.) applied to inputs before the model.

**Required abstract methods:**

- `create_gds_file(name, output_path, params) -> str` — Generates a GDS layout file from geometry parameters. Returns the path to the created file.
- `get_model(hyperparameters: dict) -> nn.Module` — Returns a PyTorch model instance configured from the given hyperparameters. Called during `ModelTrainer` hyperparameter search.
- `get_hyperparameter_search_space() -> dict` — Returns an Optuna hyperparameter search space dictionary. Used by `ModelTrainer` to tune the model.

**Optional override:**

- `postprocess_outputs(output, frequency_points)` — Post-processes raw ONNX model outputs after inference. Default implementation returns outputs unchanged.

Example:

```python
@dataclass
class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape.
    """

    name: str = "tf_octa_c_ports"
    stackup_xml: str = os.path.join(os.path.dirname(__file__), "SG13G2_nosub.xml")
    simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")
    input_parameter_iterator: InputParameterIterator = InputParameterIterator(
        picking_strategy="random",
        frequency=[1e8, 500e8],  # 1 GHz to 500 GHz
        bottom_winding_diameter=[x / 10 for x in range(200, 1201, 1)],  # 20.0 to 120.0 in 0.1 steps
        top_winding_diameter=[x / 10 for x in range(200, 1201, 1)],  # 20.0 to 120.0 in 0.1 steps
        center_displacement=[x / 10 for x in range(0, 151, 1)],   # 0.0 to 15.0 in 0.1 steps
        bottom_linewidth=[x / 10 for x in range(20, 121, 1)],     # 2.0 to 12.0 in 0.1 steps
        top_linewidth=[x / 10 for x in range(20, 121, 1)],        # 2.0 to 12.0 in 0.1 steps
    )
    features = FeatureTransformPipeline(
        # Add RatioFeature or ChebyshevFeature transforms here if desired
        # RatioFeature(i=0, j=1),         # bottom_winding_diameter / top_winding_diameter
        # ChebyshevFeature(i=5, degree=3), # Chebyshev features of frequency
    )
    dataset: BaseDataset = GeoToSParamDatasetSingleFrequency(
        n_ports=6,
        features=features,
        input_normalizer=OutputMinMaxNormalizer(),
        output_normalizer=StandardNormalizer(),
    )

    def get_hyperparameter_search_space(self) -> dict:
        return {
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "batch_size": [32, 64, 128, 256, 512],
            "epochs": optuna.distributions.IntDistribution(5, 50, step=5),
            "num_layers": optuna.distributions.IntDistribution(3, 9),
            "hidden_size": optuna.distributions.IntDistribution(128, 2048, step=128),
            "activation_function": ["GELU", "SiLU"],
        }

    def get_model(self, hyperparameters: dict) -> nn.Module:
        hidden_channels = [hyperparameters["hidden_size"]] * hyperparameters["num_layers"] + [72]
        return torchvision.ops.MLP(
            in_channels=5 + 1,  # 5 geometry params + 1 frequency
            hidden_channels=hidden_channels,
            activation_layer=getattr(nn, hyperparameters["activation_function"]),
        )

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:
        # 
        # < Put your actual GDS generation code here >
        #
        c.write_gds(output_path, with_metadata=False)
        return output_path

    # Example for a postprocessing method that plots the results and saves a Touchstone file
    def postprocess_outputs(self, output, frequency_points=None):
        """
        Converts model outputs (Re/Im) into a .sNp Touchstone file format.
        Plots the S-parameters for visualization.

        Parameters
        ----------
        output : dict
            Dictionary containing S-parameters split into real and imaginary parts.
            Example keys: 'S11_real', 'S11_imag', ..., 'SNN_real', 'SNN_imag'.
            Each value is a 1D array of length equal to len(f).
        f : array-like
            1D array of frequencies corresponding to the S-parameters.
        filename : str, optional
            Name of the Touchstone file to save, default "output.sNp".
        """
        # Frequency points are just from 1 to 200 in 1 GHz steps
        if frequency_points is None:
            frequency_points = np.arange(1, 201)  # 1 GHz to 200 GHz
        N, ntwk, output_dict = s_param_dict_to_network(output, frequency_points)
        filename = f"{self.name}.s{N}p"
        ntwk.write_touchstone(filename)

        # N, ntwk = single_ended_to_mixed_mode(ntwk)
        plot_rfic_transformer_metrics(ntwk)
        # plot_diff_s_params_and_k(ntwk)

        # Write Touchstone
        print(f"Touchstone file saved as {filename}")

        return output_dict
```

### Reference: InputParameterIterator

`InputParameterIterator` defines the set of geometry parameters and how they are sampled during GDS generation.

```python
InputParameterIterator(
    picking_strategy="random",  # "grid" / "uniform_grid", "step_grid", or "random"
    frequency=[1e8, 500e8],     # Optional: frequency range included for normalisation (not iterated)
    param_a=[...],              # List/range of possible values for each geometry parameter
    param_b=[...],
)
```

Picking strategies:

| Strategy | Behaviour |
|---|---|
| `"grid"` / `"uniform_grid"` | Uniform grid across all parameter combinations |
| `"step_grid"` | Grid using explicit step sizes |
| `"random"` | Random sampling without replacement |

### Reference: Dataset Types

| Class | Description |
|---|---|
| `GeoToSParamDatasetSingleFrequency` | One training sample per frequency point per geometry (recommended) |
| `GeoToSParamDataset` | One training sample per geometry (full frequency sweep as a vector) |

### Reference: Feature Transforms

Feature transforms are applied to geometry inputs before the model and baked into the exported ONNX.

| Class | Description |
|---|---|
| `RatioFeature(i, j)` | Appends `params[i] / params[j]` as an extra input feature |
| `ChebyshevFeature(i, degree)` | Appends Chebyshev polynomial features of `params[i]` up to `degree` |

### Reference: Normalizers

**Input normalizers** (passed as `input_normalizer` to the dataset):

| Class | Description |
|---|---|
| `OutputMinMaxNormalizer` | Min-max normalisation derived from output statistics |

**Output normalizers** (passed as `output_normalizer` to the dataset):

| Class | Description |
|---|---|
| `StandardNormalizer` | Z-score normalisation (zero mean, unit variance) |
| `OutputMinMaxNormalizer` | Min-max normalisation |

### Stackup XML File
There are multiple examples of stackup XML files in the `src/orca/geometry/examples/` directory. Often these are sufficient to get started. You can also create your own stackup XML file or adjust [one of the examples here](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/tree/main/workflow) to fit your needs.

Example:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
  <Stackup schemaVersion="2.0">
    <Materials>
      <Material Name="Activ" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="357141.0" Color="00ff00"/>
      <Material Name="Metal1" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="21640000.0" Color="39bfff"/>
      <Material Name="Metal2" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="23190000.0" Color="ccccd9"/>
      <Material Name="TopMetal1" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="27800000.0" Color="ffe6bf"/>
      <Material Name="TopMetal2" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="30300000.0" Color="ff8000"/>
      <Material Name="TopVia1" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="2191000.0" Color="ffe6bf"/>
      <Material Name="Via2" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="1660000.0" Color="ff3736"/>
      <Material Name="Via1" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="1660000.0" Color="ccccff"/>
      <Material Name="Cont" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="2390000.0" Color="00ffff"/>
      <Material Name="Passive" Type="Dielectric" Permittivity="6.6" DielectricLossTangent="0.0" Conductivity="0" Color="a0a0f0"/>
      <Material Name="SiO2" Type="Dielectric" Permittivity="4.1" DielectricLossTangent="0.0" Conductivity="0" Color="fffcad"/>
      <Material Name="AIR" Type="Dielectric" Permittivity="1.0" DielectricLossTangent="0.0" Conductivity="0" Color="d0d0d0"/>
      <Material Name="Vmim" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="2191000.0" Color="ffe6bf"/>
      <Material Name="MIM" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="500000.0" Color="e6ffbf"/>
      <Material Name="LOWLOSS" Type="Conductor" Permittivity="1" DielectricLossTangent="0" Conductivity="1E10" Color="ff0000"/>
    </Materials>
    <ELayers LengthUnit="um">
      <Dielectrics>
        <Dielectric Name="AIR" Material="AIR" Thickness="200.0000" />
        <Dielectric Name="Passive" Material="Passive" Thickness="0.4000" />
        <Dielectric Name="SiO2" Material="SiO2" Thickness="15.7303" />
        <Dielectric Name="EPI" Material="EPI" Thickness="3.7500" />
        <Dielectric Name="Substrate" Material="Substrate" Thickness="180.0000" />
      </Dielectrics>
      <Layers>
        <Substrate Offset="183.75"/>
        <Layer Name="Activ" Type="conductor" Zmin="0.0000" Zmax="0.4000" Material="Activ" Layer="1" />
        <Layer Name="Metal1" Type="conductor" Zmin="1.0400" Zmax="1.4600" Material="Metal1" Layer="8" />
        <Layer Name="Metal2" Type="conductor" Zmin="2.0000" Zmax="2.4900" Material="Metal2" Layer="10" />
        <Layer Name="TopMetal1" Type="conductor" Zmin="6.4303" Zmax="8.4303" Material="TopMetal1" Layer="126" />
        <Layer Name="TopMetal2" Type="conductor" Zmin="11.2303" Zmax="14.2303" Material="TopMetal2" Layer="134" />
        <Layer Name="SUBGND" Type="conductor" Zmin="-3.75" Zmax="0" Material="LOWLOSS" Layer="250" />
        <Layer Name="BACKSIDEGND" Type="conductor" Zmin="-190" Zmax="-183.75" Material="LOWLOSS" Layer="251" />
        <Layer Name="MIM" Type="conductor" Zmin="5.6043" Zmax="5.7540" Material="MIM" Layer="36" />
        <Layer Name="Vmim" Type="via" Zmin="5.7540" Zmax="6.4303" Material="Vmim" Layer="129" />
        <Layer Name="LBE" Type="dielectric" Zmin="-183.75" Zmax="0" Material="Air" Layer="157" />
      </Layers>
    </ELayers>
  </Stackup>
```

### Simulation Configuration File
The simulation configuration file (.simcfg) defines how to mesh and run the electromagnetic simulation in Palace. 
You can either tweak existing .simcfg files from the `src/orca/geometry/examples/` directory, create your own from scratch,
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
