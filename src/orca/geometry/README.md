## Geometry
This folder contains scripts for generating geometries for ORCA simulations. It provides a base class to allow easy creation and manipulation of different geometric shapes and configurations used in simulations.
### Layers
[`layers.py`](layers.py) holds the IHP SG13G2 layer map as plain `(layer, datatype)` tuples, e.g. `SG13G2.TopMetal2 == (134, 0)`. Use these in new geometry cells instead of importing a gdsfactory PDK package: they work with gdsfactory, gdspy and klayout alike and keep heavy PDK dependencies (sax/JAX) out of the process. `SG13G2.with_purpose(layer, PURPOSE_PIN)` gives the other datatypes of a layer.
