## Running ORCA

If you have followed the installation instructions (see [Setup](/docs/setup.md)), you are ready to run ORCA. There are currently three ways to run ORCA:

### 1. Graphical User Interface (GUI)

For an interactive experience with real-time progress visualization:

```bash
orca
```

### 2. Python Script

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

# Setup the pipeline stages
orca_instance = ORCA(
    [
        orca.GDSGenerator(num_samples=1000),
        orca.GDSConverter(),
        orca.PalaceSimulator(palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace"),
        orca.ModelTrainer(),
        orca.OnnxExporter(),
        orca.ModelTester(),
    ]
)

# Run the ORCA pipeline
orca_instance.run(geometry=geometry, cpu_cores=16)
```

This will create 1000 differently parameterized instances of the `tf_octa_c_ports` transformer geometry, run electromagnetic simulations using Palace, store the results in Touchstone format, train a machine learning model, export it to ONNX format, and finally test the model.
If you only want to run specific stages of the pipeline, you can modify the list of stages passed to the `ORCA` constructor.
Note that some stages may require additional parameters or depend on the outputs of previous stages.

### 3. OpenStack VM REST API
OpenStack is an open-source cloud computing platform (like AWS), perfect for running large-scale simulations. We provide an OpenStack VM image and an API with the corresponding client CLI in our [ORCA-OpenStack repository](https://github.com/DavidL-11/ORCA-OpenStack).