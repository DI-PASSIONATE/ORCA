## Running ORCA

If you have followed the installation instructions (see [Setup](/docs/setup.md)), you are ready to run ORCA. There are three ways to run ORCA:

### 1. Graphical User Interface (GUI)

For an interactive experience with real-time progress visualization:

```bash
orca
```

### 3. Python API

For direct integration into scripts or custom workflows:

```python
from orca import ORCA

### Example of using a custom geometry
# geometry = MyCustomGeometry( # Python class that inherits from BaseGeometry
#     name = "my_geometry",
#     stackup_xml = "/path/to/stackup.xml", # XML file defining the physical layer stackup
#     simconfig_filename = "/path/to/simconfig.simcfg" # Simulation configuration file generated manually or with setupEM
# )

# Use predefined geometry from examples
from orca.geometry.examples.transformer.tf_octa_c_ports import TransformerOcta
geometry = TransformerOcta()

orca_instance = ORCA(geometry)

orca_instance.run(num_samples=3, cpu_cores=16, palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace")
```

This will create 3 differently parameterized instances of the `tf_octa_c_ports` transformer geometry, run electromagnetic simulations using Palace, and store the results in Touchstone format.

**Note:** Model training and evaluation are not yet implemented.