from dataclasses import dataclass, field
import threading
from itertools import product
import numpy as np

class InputParameterIterator:
    """
    Class for keeping track of geometry input parameters.
    
    Attributes:
        input_values (dict): Dictionary mapping parameter names to their values or ranges.
    """

    def __init__(self, picking_strategy: str = "grid", **input_values):
        """
        Initializes InputParameters with given input values and picking strategy.   

        Args:
            picking_strategy (str): Strategy for picking parameters ('grid', 'random', etc.).
            **input_values: Keyword arguments representing parameter names and their possible ranges, e.g. radius_bottom=range(20, 100). Possible values can be lists, ranges, or numpy arrays.
        """
        # Check if all input_values are lists or ranges
        for name, values in input_values.items():
            print(f"Input parameter '{name}' has type {type(values)} with values: {values}")
            if not isinstance(values, (list, range, np.ndarray)):
                raise ValueError(f"Input parameter '{name}' must be a list, range, or numpy array of possible values.")

        self.picking_strategy = picking_strategy
        self.n_inputs = len(input_values)
        self.input_values = input_values
        self.input_names = list(input_values.keys())
        
        self._lock = threading.Lock() # For thread-safe iteration

        if self.picking_strategy == "grid":
            self._iterator = product(*(input_values[name] for name in self.input_names))
        else:
            raise ValueError(f"Unsupported picking strategy: {self.picking_strategy}. Supported strategies are 'grid'.")
        

    def __iter__(self):
        self.n_geometries_created = 0
        return self
    
    def __next__(self) -> dict[str, any]:
        with self._lock: # Ensure thread-safe access
            self.n_geometries_created += 1 # May be used for logging or tracking

            if self.picking_strategy == "grid":
                try:
                    params = next(self._iterator) # Raises StopIteration when exhausted, which is propagated to this iterator
                    return dict(zip(self.input_names, params))
                except StopIteration:
                    raise
            else:
                raise ValueError(f"Unsupported picking strategy: {self.picking_strategy}. Supported strategies are 'grid'.")
