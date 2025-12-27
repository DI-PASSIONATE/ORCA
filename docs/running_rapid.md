## Running RAPID

If you have followed the installation instructions (see [Setup](/docs/setup.md)), you are ready to run RAPID. Below is a simple example of how to use RAPID with a predefined geometry.

```python
from rapid import RAPID
from rapid.geometry.examples.transformer.tf_octa_c_ports import Transformer

geometry = Transformer(
    name="tf_octa_c_ports",
    stackup_xml="/home/david/Documents/git/RAPID/src/rapid/geometry/examples/transformer/tf_octa_c_ports.xml",
    simconfig_filename="/home/david/Documents/git/RAPID/src/rapid/geometry/examples/transformer/tf_octa_c_ports.simcfg",
)

rapid = RAPID(geometry)

rapid.run(num_samples=3, cpu_cores=12)
```

This will create 3 differently parameterized instances of the `tf_octa_c_ports` transformer geometry, run electromagnetic simulations using Palace, store the results in Touchstone format. TODO: Train a model using the generated data.