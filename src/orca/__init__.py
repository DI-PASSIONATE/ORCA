# Import stuff here so that they are available at the package level (i.e. from orca import ORCA, BaseGeometry, InputParameters)

MIN_FREQUENCY = 1e9  # 1 GHz
MAX_FREQUENCY = 200e9  # 200 GHz

from .orca import ORCA
from .geometry.base_geometry import BaseGeometry
from .geometry.input_parameters import InputParameterIterator