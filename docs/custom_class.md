## Custom Classes
If you want to simulate and predict your own geometry, you simply need to create three files:

- A python file extending the `BaseGeometry` class and implementing the required methods.
- A stackup XML file defining the layers of your geometry.
- A simulation configuration file defining the simulation parameters for Palace.

Once you have these three files, you can head over to [Running ORCA](/docs/running_orca.md) to see how to run ORCA with these files.

### Python Class
The Python class should extend the `orca.BaseGeometry` class and implement the following methods:

- `__init__(self, name: str, stackup_xml: str, simconfig_filename: str)`: You can define the number of input and output parameters here. Make sure to call the superclass constructor
- `create_gds_file(self) -> str`: This method should create the GDS file (e.g. using [gdsfactory](https://gdsfactory.github.io/gdsfactory/)) for your geometry and return the path to the created file.
- `get_input_parameters(self) -> InputParameterIterator`: This method should return the next set of input parameters for parameterizing the geometry.

Example:

```python
from orca import BaseGeometry
from orca.geometry.input_parameters import InputParameterIterator
from ihp import PDK


class TransformerOcta(BaseGeometry):
    """
    Represents a transformer geometry with octagonal shape and C-ports.
    No input parameters are required for this geometry.
    """

    def __init__(self,
                 name = "tf_octa_c_ports",
                 stackup_xml: str = os.path.join(os.path.dirname(__file__), "..", "SG13G2_nosub.xml"),
                 simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg"),
                 params = None
                ):

        # Call the base class constructor with the parameters
        super().__init__(name, stackup_xml, simconfig_filename, params)
    
    def get_input_parameters(self) -> InputParameterIterator:
        return InputParameterIterator(
            picking_strategy="grid",
            input_winding_diameter = range(60, 101, 10), # 20, 101, 5
            output_winding_diameter = range(60, 101, 10), # 20, 101, 5
            center_displacement = range(0, 21, 10), # 0, 21, 1
            bottom_linewidth = range(5, 9, 3), # 2, 9, 1
            upper_linewidth = range(5, 9, 3), # 2, 9, 1
        )

    def create_gds_file(self, params: dict[str, any]) -> str:
        output_path = os.path.join(
            os.getcwd(),
            "geometries",
            self.name + ".gds"
        )

        # create component here
        c.write_gds(output_path, with_metadata=False)
        return output_path
```

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
